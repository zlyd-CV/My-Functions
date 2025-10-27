from matplotlib import pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import random

class DynamicTrainingPlot:
    """
    使用示例：
    dynamic_plot = DynamicTrainingPlot(
        total_epochs=epochs,
        title = "这是示例"
    )

    在每个epoch的外循环末尾：
    dynamic_plot.update(
        epoch=epoch,
        train_loss=avg_train_loss,
        test_loss=avg_test_loss,
        train_acc=train_acc,
        test_acc=test_acc
    )

    训练结束时：
    dynamic_plot.close()
    """

    # 动态训练指标可视化类，用于实时展示训练/测试的准确率和损失曲线"""
    def __init__(self, total_epochs, title="请指定图表标题"):
        # 初始化初始化动态图表，用于存储历史数据
        self.total_epochs = total_epochs  # 总训练轮次
        self.train_loss_history = []  # 训练损失历史
        self.test_loss_history = []  # 测试损失历史
        self.train_acc_history = []  # 训练准确率历史
        self.test_acc_history = []  # 测试准确率历史

        # 设置中文字体
        plt.rcParams["font.family"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        # 启用交互模式
        plt.ion()

        # 创建图表和轴对象
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        # fig:包含了所有绘图元素/ax:是实际绘制数据的地方
        self.ax.set_title(title)
        self.ax.set_xlabel("epoch")
        self.ax.set_ylabel("acc/loss")
        self.ax.set_xlim(1, total_epochs)  # x轴范围固定为总迭代次数
        self.ax.set_ylim(0, 1.2)  # y轴范围（适合acc和loss，loss可能超过1时需调整）
        self.ax.grid(True, linestyle="--", alpha=0.5)  # 给图表背景添加网格线。方便观察

        # 初始化四条曲线（线对象用于后续更新）
        self.train_loss_line, = self.ax.plot([], [], color='red', label="训练损失", linewidth=2)
        self.test_loss_line, = self.ax.plot([], [], color='orange', label="测试损失", linewidth=2)
        self.train_acc_line, = self.ax.plot([], [], color='green', label="训练准确率", linewidth=2, linestyle="--")
        self.test_acc_line, = self.ax.plot([], [], color='blue', label="测试准确率", linewidth=2, linestyle="--")

        # 添加图例和自动布局调整
        self.ax.legend()
        plt.tight_layout()

    def update(self, epoch, train_loss, test_loss, train_acc, test_acc):
        """
        更新图表数据

        参数:
        :param epoch: 当前轮次（从1开始），确保训练循环为 for epoch in range(1,epochs+1)
        :param train_loss: 当前训练损失
        :param test_loss: 当前测试损失
        :param train_acc: 当前训练准确率
        :param test_acc: 当前测试准确率

        """
        # 存储当前轮次的指标
        self.train_loss_history.append(train_loss)
        self.test_loss_history.append(test_loss)
        self.train_acc_history.append(train_acc)
        self.test_acc_history.append(test_acc)

        # 生成x轴数据（已训练的轮次）
        x = list(range(1, epoch + 1))

        # set_data(x, y)是曲线对象（Line2D）的一个方法，作用是用新的 x 和 y 数据更新这条曲线
        self.train_loss_line.set_data(x, self.train_loss_history)
        self.test_loss_line.set_data(x, self.test_loss_history)
        self.train_acc_line.set_data(x, self.train_acc_history)
        self.test_acc_line.set_data(x, self.test_acc_history)

        # 高效刷新图表
        self.ax.draw_artist(self.ax.patch)  # 重绘背景
        for line in [self.train_loss_line, self.test_loss_line,
                     self.train_acc_line, self.test_acc_line]:
            self.ax.draw_artist(line)
        self.fig.canvas.flush_events()  # 刷新画布

    def close(self, save_plt=False):
        """
        关闭交互模式，保存图表（可选）
        """
        plt.ioff()  # 关闭交互模式
        if save_plt:
            plt.savefig("training_curve.png")  # 可选：保存最终图表
        plt.show()  # 显示最终图表（阻塞程序，直到关闭窗口）

def generate_tree(path, indent=""):
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        print(f"{indent}├── {item}")
        if os.path.isdir(full_path):
            generate_tree(full_path, indent + "│   ")
