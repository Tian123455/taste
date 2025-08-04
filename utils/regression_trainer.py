import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from utils.trainer import Trainer
from datasets.dataset import TwoStreamBatchSampler
from torch.utils.data import DataLoader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(message)s')
logger = logging.getLogger(__name__)


class Reg_Trainer(Trainer):
    """人群计数回归训练器（显存优化版）"""

    def __init__(self, args):
        super().__init__(args)
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.inpaint_prompts = []
        self.gradient_accumulation = args.gradient_accumulation  # 梯度累积步数

    def set_model(self, model):
        """设置模型"""
        self.model = model.to(self.device)

    def set_datasets(self, train_dataset, val_dataset=None):
        """设置数据集"""
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_count = train_dataset.label_count if hasattr(train_dataset, 'label_count') else {}

    def setup(self):
        """设置训练环境"""
        if self.model is None:
            raise ValueError("请先调用set_model设置模型")
        if self.train_dataset is None:
            raise ValueError("请先调用set_datasets设置训练数据集")

        # 获取有标签和无标签数据索引
        labeled_idxs = np.where(self.train_dataset.labeled_mask)[0]
        unlabeled_idxs = np.where(~self.train_dataset.labeled_mask)[0]

        # 日志输出数据分布
        logger.info(f"训练数据分布 - 有标签: {len(labeled_idxs)}, 无标签: {len(unlabeled_idxs)}")

        # 自动调整无标签批次大小（如果无标签数据不足）
        secondary_batch_size = self.args.batch_size - self.args.labeled_batch_size
        if len(unlabeled_idxs) < secondary_batch_size:
            new_secondary = max(0, len(unlabeled_idxs))
            new_primary = self.args.batch_size - new_secondary
            logger.warning(f"无标签数据不足，自动调整批次大小 - 有标签: {new_primary}, 无标签: {new_secondary}")
            self.args.labeled_batch_size = new_primary
            secondary_batch_size = new_secondary

        # 创建批次采样器
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs,
            self.args.batch_size, secondary_batch_size
        )

        # 创建训练数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True
        )

        # 创建验证数据加载器（使用更小的批次）
        val_batch_size = max(1, self.args.batch_size // 2)
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=val_batch_size,  # 验证批次减小一半
                shuffle=False,
                num_workers=min(2, self.args.num_workers),  # 减少验证时的线程
                pin_memory=True
            )

        # 初始化优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        # 初始化学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.args.lr_step,
            gamma=self.args.lr_gamma
        )

        # 处理提示词文件
        self.inpaint_prompts = ["a scene with people"]  # 默认提示词
        if self.args.inpaint_prompts_file and os.path.exists(self.args.inpaint_prompts_file):
            try:
                with open(self.args.inpaint_prompts_file, "r") as f:
                    self.inpaint_prompts = [line.strip() for line in f if line.strip()]
                logger.info(f"加载 {len(self.inpaint_prompts)} 条提示词")
            except Exception as e:
                logger.warning(f"提示词文件加载失败: {e}，使用默认值")

    def train_one_epoch(self, epoch):
        """训练一个epoch（使用梯度累积）"""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()  # 初始化梯度

        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            data, target = data.to(self.device), target.to(self.device)

            # 前向传播
            output = self.model(data)
            loss = F.mse_loss(output, target)

            # 梯度累积（将损失除以累积步数）
            loss = loss / self.gradient_accumulation
            loss.backward()  # 计算梯度但不更新参数

            total_loss += loss.item() * self.gradient_accumulation  # 恢复实际损失值

            # 每累积一定步数后更新参数
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                self.optimizer.step()  # 更新参数
                self.optimizer.zero_grad()  # 重置梯度

            # 打印批次损失
            if batch_idx % self.args.log_interval == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item() * self.gradient_accumulation:.6f}")

        # 处理剩余未更新的梯度
        if (len(self.train_loader) % self.gradient_accumulation) != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / len(self.train_loader)
        logger.info(f"Epoch {epoch} 平均训练损失: {avg_loss:.6f}")
        return avg_loss

    def validate(self):
        """验证模型（使用更小的批次）"""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():  # 关闭梯度计算以节省显存
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += F.mse_loss(output, target).item()

        avg_val_loss = val_loss / len(self.val_loader)
        logger.info(f"验证损失: {avg_val_loss:.6f}")
        return avg_val_loss

    def train(self):
        """完整训练流程"""
        best_val_loss = float('inf')

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_one_epoch(epoch)

            if self.val_loader:
                val_loss = self.validate()

                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(f"best_model_epoch_{epoch}.pth")
                    logger.info(f"保存最佳模型 (验证损失: {best_val_loss:.6f})")

            # 更新学习率
            self.scheduler.step()

        # 保存最终模型
        self.save_model("final_model.pth")
        logger.info("训练完成")
