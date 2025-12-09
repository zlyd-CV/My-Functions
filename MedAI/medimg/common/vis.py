"""
File: vis.py
Description: 医学影像可视化工具（如切片可视化）。
Author: zlyd-CV
License: MIT
"""
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_slices(image_3d, max_height=20, save_path=None):
    """
    :param image_3d: 三维图像数据的ndarray数组,形状为 (D, H, W)
    :param max_height: 图像显示的最大高度，防止图像过高
    :param save_path: 保存路径（可选，目录或文件路径）
    :return: 显示三维图像沿Z轴的切片图像
    """
    if image_3d.ndim != 3:
        raise ValueError("输入图像必须是三维数组，形状为 (D, H, W)")

    depth, h, w = image_3d.shape
    n_cols = int(np.ceil(np.sqrt(depth)))
    n_rows = int(np.ceil(depth / n_cols))

    # 调整图像大小，防止过高
    figsize = (n_cols * 2, min(max_height, n_rows * 2 * h / w))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # 统一处理axes为一维数组，方便遍历，解决单张或单行单列时的索引问题
    axes = np.atleast_1d(axes).flatten()

    for i, ax in enumerate(axes):
        if i < depth:
            ax.imshow(image_3d[i], cmap='gray')
            ax.set_title(f'{i+1}/{depth}')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        # 如果路径不以常见图片格式结尾，视为目录
        if not save_path.lower().endswith(('.png', '.jpg', '.jpeg', 'tiff')):
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "slices.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"切片图像已保存: {save_path}")

    plt.show()
