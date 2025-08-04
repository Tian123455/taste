import os
import torch


class Trainer:
    """基础训练器类（修复后版本）"""

    def __init__(self, args):
        self.args = args
        self.device = args.device

        # 移除对args.content的依赖，使用save_dir作为模型保存根目录
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_model(self, model_name):
        """保存模型到指定目录"""
        save_path = os.path.join(self.save_dir, model_name)
        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存至: {save_path}")

    def load_model(self, model_path):
        """加载模型"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"已加载模型: {model_path}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
