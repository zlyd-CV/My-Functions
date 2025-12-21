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
    def __init__(self, metrics: List[str]):
        """
        初始化日志记录器
        :param metrics: 指标列表 (必须包含 'epoch')
        使用示例：
        logger = TrainingLogger(metrics=["epoch", "loss", "accuracy"])
        logger.loss.append(0.5)
        getattr(logger, 'accuracy').append(0.8)
        logger.save_csv(save_dir="./logs", experiment_name="exp1", model_type="CNN", comment="best")
        logger.plot(save_dir="./logs", experiment_name="exp1", model_type="CNN")
        """
        if 'epoch' not in metrics:
            raise ValueError("metrics must include 'epoch'")
        self.metrics = metrics
        for m in metrics:
            setattr(self, m, [])  # 动态创建属性列表

    def save_csv(self, save_dir: str, experiment_name: str, model_type: str, comment=''):
        """保存数据到CSV"""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{experiment_name}_{model_type}")
        others = [m for m in self.metrics if m != 'epoch']
        with open(f"{save_path}.csv", 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['epoch', 'experiment_name', 'model_type'] +
                       others + (['comment'] if comment else []))
            w.writerows([e, experiment_name, model_type, *o] + ([comment] if comment else [])
                        for e, *o in zip(self.epoch, *[getattr(self, m) for m in others]))
        print(f"Saved: {save_path}.csv")

    def plot(self, save_dir: str, experiment_name: str, model_type: str):
        """绘制并保存曲线"""
        save_path = os.path.join(save_dir, f"{experiment_name}_{model_type}")
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        for m in self.metrics:
            if m.lower() != 'epoch':  # 排除epoch作为y轴
                plt.plot(getattr(self, m), label=m)

        plt.title(f"{os.path.basename(save_path)}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_path}.png", dpi=300)
        plt.close()
        print(f"Plot saved: {save_path}.png")
