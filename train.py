import argparse
import os
import torch
from utils.regression_trainer import Reg_Trainer
from model.model import mamba
from datasets.dataset import Crowd


def parse_args():
    parser = argparse.ArgumentParser(description="人群计数模型训练（显存优化版）")
    # 数据参数 - 减小默认图像尺寸和批次大小
    parser.add_argument("--data-dir",
                        type=str,
                        default="/home/tian/桌面/taste_more_taste_better-main/data_dir/ShanghaiTech/part_A_final",
                        help="数据集根目录")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小（减小以节省显存）")
    parser.add_argument("--labeled-batch-size", type=int, default=1, help="有标签批次大小")
    parser.add_argument("--num-workers", type=int, default=2, help="数据加载线程数（减小以节省内存）")
    parser.add_argument("--img-height", type=int, default=256, help="图像高度（减小以节省显存）")
    parser.add_argument("--img-width", type=int, default=256, help="图像宽度（减小以节省显存）")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=50, help="训练轮次")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率（随批次减小而降低）")
    parser.add_argument("--lr-step", type=int, default=10, help="学习率衰减步长")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="衰减系数")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--log-interval", type=int, default=10, help="日志间隔")

    # 其他参数 - 添加显存优化选项
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inpaint-prompts-file", type=str, default="", help="提示词文件")
    parser.add_argument("--save-dir", type=str, default="./models", help="模型保存目录")
    parser.add_argument("--gradient-accumulation", type=int, default=4, help="梯度累积步数（模拟大批次）")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 打印当前配置
    logger.info(f"使用数据集路径: {args.data_dir}")
    logger.info(f"图像尺寸: {args.img_height}x{args.img_width}")
    logger.info(f"批次大小: {args.batch_size}, 梯度累积: {args.gradient_accumulation}")
    logger.info(f"训练设备: {args.device}")

    # 1. 初始化训练器
    trainer = Reg_Trainer(args)

    # 2. 设置模型（传递图像尺寸参数给模型）
    model = mamba(img_height=args.img_height, img_width=args.img_width)
    trainer.set_model(model)
    print("模型初始化完成")

    # 3. 设置数据集（传递图像尺寸参数）
    img_size = (args.img_height, args.img_width)
    train_data_path = os.path.join(args.data_dir, "train")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"训练集目录不存在: {train_data_path}")

    train_dataset = Crowd(train_data_path, img_size=img_size)
    print(f"训练集加载完成，共 {len(train_dataset)} 张图像")

    val_dataset = None
    val_data_path = os.path.join(args.data_dir, "val")
    if os.path.exists(val_data_path):
        try:
            val_dataset = Crowd(val_data_path, img_size=img_size)
            print(f"验证集加载完成，共 {len(val_dataset)} 张图像")
        except Exception as e:
            print(f"验证集加载失败: {e}")

    trainer.set_datasets(train_dataset, val_dataset)

    # 4. 开始训练
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    # 初始化日志
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    main()
