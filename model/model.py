import torch
import torch.nn as nn
import torch.nn.functional as F


class mamba(nn.Module):
    """轻量化人群计数模型（适合小显存GPU）"""

    def __init__(self, img_height=256, img_width=256, in_channels=3):
        super(mamba, self).__init__()
        self.img_height = img_height
        self.img_width = img_width

        # 轻量化特征提取网络（减少通道数）
        self.features = nn.Sequential(
            # 输入: (3, H, W)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/2

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 1/8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(inplace=True)
        )

        # 轻量化密度图预测头
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 减少通道数
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1)  # 输出单通道密度图
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 特征提取
        x = self.features(x)

        # 密度图预测
        density_map = self.head(x)

        # 确保输出密度图尺寸是输入图像的1/4
        target_size = (self.img_height // 4, self.img_width // 4)
        density_map = F.interpolate(
            density_map,
            size=target_size,
            mode='bicubic',
            align_corners=False
        )

        return density_map
