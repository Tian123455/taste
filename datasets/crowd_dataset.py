from PIL import Image
import numpy as np
import os
import shutil
from glob import glob
import argparse
import scipy.spatial
import tqdm
import scipy.ndimage as ndimage
import h5py
import random


def gaussian_filter_density(gt):
    """生成高斯滤波密度图"""
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, _ = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            if sigma > 15:
                sigma = 15
        else:
            sigma = np.average(np.array(gt.shape)) / 4.0
        density += ndimage.filters.gaussian_filter(pt2d, sigma, mode="constant")
    return density


def cal_new_size(im_h, im_w, min_size, max_size):
    """调整图像尺寸"""
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
            if im_h < min_size:
                ratio = 1.0 * min_size / im_h
                im_h = min_size
                im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
            if im_w < min_size:
                ratio = 1.0 * min_size / im_w
                im_w = min_size
                im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def generate_data(im_path, min_size, max_size, overwrite_image=True):
    """处理单张图像和标注，不生成重复图像"""
    im = Image.open(im_path)
    im_w, im_h = im.size  # (宽, 高)

    # 解析标注文件路径
    if 'train' in im_path:
        gt_dir = 'gt_train'
    elif 'val' in im_path:
        gt_dir = 'gt_val'
    else:
        gt_dir = 'gt_test'
    mat_path = im_path.replace('images', gt_dir).replace('.jpg', '.txt')

    # 读取标注点
    points = []
    if os.path.exists(mat_path):
        with open(mat_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                x, y = map(float, line.split())
                points.append([x, y])
    points = np.array(points)

    # 过滤超出图像范围的点
    if len(points) > 0:
        idx_mask = (
                (points[:, 0] >= 0) &
                (points[:, 0] <= im_w) &
                (points[:, 1] >= 0) &
                (points[:, 1] <= im_h)
        )
        points = points[idx_mask]

    # 调整图像尺寸
    im_h_new, im_w_new, ratio = cal_new_size(im_h, im_w, min_size, max_size)
    if ratio != 1.0:
        im = im.resize((im_w_new, im_h_new), Image.LANCZOS)
        points = points * ratio  # 按比例调整点坐标

        # 如果需要覆盖原图
        if overwrite_image:
            im.save(im_path)

    return im, points


def create_val_set(train_dir, val_ratio=0.2, random_seed=42):
    """从训练集中划分一部分作为验证集"""
    # 设置随机种子，确保结果可复现
    random.seed(random_seed)

    # 创建验证集目录
    val_dir = os.path.dirname(train_dir) + "/val"
    val_img_dir = os.path.join(val_dir, "images")
    val_gt_dir = os.path.join(val_dir, "gt_val")

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_gt_dir, exist_ok=True)

    # 获取训练集图像和标注文件
    train_img_dir = os.path.join(train_dir, "images")
    train_gt_dir = os.path.join(train_dir, "gt_train")

    img_list = glob(os.path.join(train_img_dir, "*.jpg"))
    if not img_list:
        raise FileNotFoundError(f"未在 {train_img_dir} 找到图像文件")

    # 随机选择部分图像作为验证集
    num_val = int(len(img_list) * val_ratio)
    val_imgs = random.sample(img_list, num_val)

    # 移动图像和对应的标注到验证集目录
    for img_path in tqdm.tqdm(val_imgs, desc="创建验证集"):
        img_name = os.path.basename(img_path)
        gt_name = os.path.splitext(img_name)[0] + ".txt"
        gt_path = os.path.join(train_gt_dir, gt_name)

        # 移动图像
        shutil.move(img_path, os.path.join(val_img_dir, img_name))

        # 移动标注文件
        if os.path.exists(gt_path):
            shutil.move(gt_path, os.path.join(val_gt_dir, gt_name))

    print(f"已从训练集中划分 {num_val} 张图像到验证集")
    return val_dir


def parse_args():
    parser = argparse.ArgumentParser(description="ShanghaiTech数据集预处理并创建验证集")
    parser.add_argument(
        "--base-dir", default='/home/tian/桌面/taste_more_taste_better-main/data_dir/ShanghaiTech/part_A_final',
        help="数据集目录")
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="从训练集中划分验证集的比例，默认0.2(20%)"
    )
    parser.add_argument(
        "--min-size", type=int, default=512, help="图像最小尺寸"
    )
    parser.add_argument(
        "--max-size", type=int, default=1920, help="图像最大尺寸"
    )
    parser.add_argument(
        "--no-overwrite", action="store_true",
        help="不覆盖原始图像，只调整标注点坐标"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_dir = args.base_dir
    val_ratio = args.val_ratio
    min_size = args.min_size
    max_size = args.max_size
    overwrite_image = not args.no_overwrite

    # 1. 从训练集划分验证集
    train_dir = os.path.join(base_dir, "train")
    val_dir = create_val_set(train_dir, val_ratio)

    # 2. 处理所有数据集（train, val, test）
    for phase in ["train", "val", "test"]:
        # 阶段目录路径
        phase_dir = os.path.join(base_dir, phase)

        # 创建所需子目录
        img_dir = os.path.join(phase_dir, "images")
        pts_dir = os.path.join(phase_dir, "gt_points")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(pts_dir, exist_ok=True)

        den_dir = None
        if phase in ["train", "val"]:  # 为训练集和验证集都生成密度图
            den_dir = os.path.join(phase_dir, "gt_den")
            os.makedirs(den_dir, exist_ok=True)

        # 获取图像列表
        im_list = sorted(glob(os.path.join(img_dir, "*.jpg")))
        if not im_list:
            print(f"警告：未在 {img_dir} 找到图像文件")
            continue

        # 处理每张图像
        for im_path in tqdm.tqdm(im_list, desc=f"处理{phase}集"):
            im, points = generate_data(im_path, min_size, max_size, overwrite_image)
            filename = os.path.basename(im_path)
            name = os.path.splitext(filename)[0]

            # 保存点坐标(npy)
            np.save(os.path.join(pts_dir, f"{name}.npy"), points)

            # 生成密度图(h5)
            if phase in ["train", "val"]:
                w, h = im.size
                gt = np.zeros((h, w), dtype=np.float32)
                for (x, y) in points:
                    x = int(round(x))
                    y = int(round(y))
                    if 0 <= x < w and 0 <= y < h:
                        gt[y, x] = 1.0
                density_map = gaussian_filter_density(gt)
                with h5py.File(os.path.join(den_dir, f"{name}.h5"), "w") as hf:
                    hf["density_map"] = density_map

    print(f"预处理完成！结果已保存至 {base_dir}")
    print(f"目录结构包含: train, val, test 三个数据集")
