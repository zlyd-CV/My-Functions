"""
File: logger.py
Description: 训练日志记录器,支持CSV保存和曲线绘制。
Author: zlyd-CV
License: MIT
"""
import os
import csv
import matplotlib.pyplot as plt
from typing import List


class TrainingLogger:
    def __init__(self, save_dir: str, experiment_name: str, model_type: str, metrics: List[str]):
        """
        初始化日志记录器
        :param save_dir: 保存目录
        :param experiment_name: 实验名称
        :param model_type: 模型类型
        :param metrics: 指标列表 (必须包含 'epoch')
        使用示例：
        logger = TrainingLogger(save_dir="./logs", experiment_name="experiment1", model_type="CNN", metrics=["epoch", "loss", "accuracy"])
        logger.loss.append(0.5)  # 像loss列表中添加数据
        logger.accuracy.append(0.8)  # 像accuracy列表中添加数据
        logger.save()  # 保存数据到CSV
        logger.plot()  # 绘制并保存曲线
        """
        if 'epoch' not in metrics:
            raise ValueError("metrics must include 'epoch'")
        self.experiment_name, self.model_type = experiment_name, model_type
        self.save_path = os.path.join(
            save_dir, f"{experiment_name}_{model_type}")
        self.metrics = metrics
        for m in metrics:
            setattr(self, m, [])
        os.makedirs(save_dir, exist_ok=True)

    def save(self):
        """保存数据到CSV"""
        others = [m for m in self.metrics if m != 'epoch']
        with open(f"{self.save_path}.csv", 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'experiment_name', 'model_type'] + others)
            w.writerows([e, self.experiment_name, self.model_type, *o]
                        for e, *o in zip(self.epoch, *[getattr(self, m) for m in others]))
        print(f"Saved: {self.save_path}.csv")

    def plot(self):
        """绘制并保存曲线"""
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        for m in self.metrics:
            if m.lower() != 'epoch':  # 排除epoch作为y轴
                plt.plot(getattr(self, m), label=m)

        plt.title(f"{os.path.basename(self.save_path)}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}.png", dpi=300)
        plt.close()
        print(f"Plot saved: {self.save_path}.png")
