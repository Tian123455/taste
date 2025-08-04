import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py
from glob import glob
from torchvision import transforms


class Crowd(Dataset):
    """人群计数数据集（统一图像尺寸）"""

    def __init__(self, data_dir, img_size=(512, 512)):  # 添加图像尺寸参数
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.im_list = sorted(glob(os.path.join(self.images_dir, "*.jpg")))
        self.img_size = img_size  # 统一的图像尺寸 (高, 宽)

        # 定义图像变换：Resize + 转换为张量 + 归一化
        self.transform = transforms.Compose([
            transforms.Resize(img_size),  # 关键：统一图像尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if not self.im_list:
            raise FileNotFoundError(f"未找到图像文件: {self.images_dir}")

        # 标签掩码（全部标记为有标签）
        self.labeled_mask = np.ones(len(self.im_list), dtype=bool)
        self.label_count = {0: len(self.im_list)}

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, index):
        # 加载图像
        img_path = self.im_list[index]
        img = Image.open(img_path).convert('RGB')

        # 加载点坐标
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        pts_path = os.path.join(self.data_dir, "gt_points", f"{img_name}.npy")
        if not os.path.exists(pts_path):
            raise FileNotFoundError(f"点坐标文件不存在: {pts_path}")
        points = np.load(pts_path)

        # 加载密度图
        den_path = os.path.join(self.data_dir, "gt_den", f"{img_name}.h5")
        if os.path.exists(den_path):
            with h5py.File(den_path, 'r') as hf:
                density_map = np.array(hf['density_map'], dtype=np.float32)
        else:
            # 创建与图像尺寸匹配的空密度图
            density_map = np.zeros((self.img_size[0] // 4, self.img_size[1] // 4), dtype=np.float32)

        # 调整密度图尺寸以匹配图像尺寸（使用双三次插值）
        density_map = self.resize_density_map(density_map, self.img_size)

        # 应用图像变换（包括Resize）
        img = self.transform(img)

        return img, torch.from_numpy(density_map).float()

    def resize_density_map(self, density_map, target_size):
        """调整密度图尺寸以匹配Resize后的图像"""
        # 密度图通常是图像尺寸的1/4
        target_den_size = (target_size[0] // 4, target_size[1] // 4)

        # 使用双三次插值调整密度图尺寸
        den_tensor = torch.from_numpy(density_map).unsqueeze(0).unsqueeze(0)
        den_resized = torch.nn.functional.interpolate(
            den_tensor,
            size=target_den_size,
            mode='bicubic',
            align_corners=False
        )

        # 保持总人数不变（缩放密度值）
        original_sum = density_map.sum()
        resized_sum = den_resized.squeeze().numpy().sum()

        if resized_sum > 0:
            den_resized = den_resized * (original_sum / resized_sum)

        return den_resized.squeeze().numpy()


class TwoStreamBatchSampler(torch.utils.data.sampler.Sampler):
    """双流批次采样器"""

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices

        # 如果没有无标签数据，将secondary_batch_size设为0
        if len(secondary_indices) == 0:
            secondary_batch_size = 0

        # 确保批次大小有效
        self.secondary_batch_size = max(0, min(secondary_batch_size, len(secondary_indices)))
        self.primary_batch_size = batch_size - self.secondary_batch_size

        # 确保主要批次大小有效
        self.primary_batch_size = max(1, min(self.primary_batch_size, len(primary_indices)))

        # 修正后的断言
        assert len(self.primary_indices) >= self.primary_batch_size > 0, \
            f"有标签数据不足，需要至少 {self.primary_batch_size} 个，实际有 {len(self.primary_indices)} 个"
        assert self.secondary_batch_size >= 0, \
            f"无标签批次大小必须非负，实际为 {self.secondary_batch_size}"
        assert (self.primary_batch_size + self.secondary_batch_size) == batch_size, \
            f"批次大小不匹配: {self.primary_batch_size} + {self.secondary_batch_size} != {batch_size}"

    def __iter__(self):
        primary_iter = self._infinite_shuffle(self.primary_indices)

        # 只有当有 secondary 数据时才创建迭代器
        if self.secondary_batch_size > 0:
            secondary_iter = self._infinite_shuffle(self.secondary_indices)

        while True:
            batch = [next(primary_iter) for _ in range(self.primary_batch_size)]

            # 只有当有 secondary 数据时才添加
            if self.secondary_batch_size > 0:
                batch += [next(secondary_iter) for _ in range(self.secondary_batch_size)]

            yield batch

    def __len__(self):
        return max(len(self.primary_indices) // self.primary_batch_size, 1)

    def _infinite_shuffle(self, indices):
        while True:
            yield from np.random.permutation(indices)
