# 本项目用于对数据集的读取与预处理
from typing import Optional, Callable, Any, Tuple
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image


# 合并图像和掩码为一个数据集(Dataset对象)，并做出预处理
class CombinedDataset(Dataset):
    """
    语义分割场景专用的「图像-掩码」数据集合并类，用于将图像数据集与对应掩码数据集对齐。

    核心功能：
    1. 自动校验图像与掩码的长度、空间尺寸一致性
    2. 支持图像与掩码的同步数据增强（保证增强操作对齐）
    3. 自动处理掩码维度/数据类型（适配语义分割标签要求）

    适用场景：
    - 二分类/多分类语义分割任务
    - 图像数据集与掩码数据集分离存储的场景
    - 需要同步增强（如翻转、裁剪）的训练流程

    注意事项：
    Dataset里的图像一般有Tensor(Pytorch数据类型)、ndarray(opencv读取)、Image(PIL读取)三种数据类型
    """

    def __init__(self, image_dataset, mask_dataset, transform_image: Optional[Callable] = None,
                 transform_mask: Optional[Callable] = None, validate_full_dataset: bool = False):
        """
        :param image_dataset: 图像数据集，传入的是Dataset类不是Dataloader类
        :param mask_dataset: 掩码数据集，适合语义分割的场景
        :param transform_image:对图像要做的预处理
        :param transform_mask:对掩码要做的预处理（一般为调整图像大小）
        :param validate_full_dataset:是否验证整个数据集的图像与掩码尺寸是否匹配
        """
        # 确保两个数据集长度一致
        if len(image_dataset) != len(mask_dataset):
            raise ValueError(f"图像数据集长度({len(image_dataset)})与掩码数据集长度({len(mask_dataset)})不匹配")

        self.image_dataset = image_dataset  # 图像Dataset类
        self.mask_dataset = mask_dataset  # 掩码Dataset类
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.validate_full_dataset = validate_full_dataset  # 是否开启形状校验

        # 检查图像与掩码的空间属性是否一致
        self.validate_spatial_consistency()

    # 确保图像是元组数据类型时不报错，只提取第一个维度数据（图像数据）
    @staticmethod
    def extract_core_data(item: Any) -> Any:
        """从子Dataset的返回值中提取核心数据（图像/掩码）：
        - 若item是元组，默认第一个元素为核心数据（忽略其他元信息）
        - 若item是单数据（Tensor/ndarray/PIL），直接返回
        """
        if isinstance(item, tuple):
            return item[0]  # 假设元组格式：(核心数据, 其他信息)
        return item

    # 获得图像的空间维度，该方法为静态方法（不带self参数）
    @staticmethod
    def get_spatial_dims(images_and_masks: Any) -> Tuple[int, int]:
        """提取对象的空间维度（高, 宽），支持PIL/ndarray/Tensor类型"""
        if isinstance(images_and_masks, torch.Tensor):
            # PyTorch张量通常为 (C, H, W) 或 (H, W)
            if len(images_and_masks.shape) == 3:
                return images_and_masks.shape[1], images_and_masks.shape[2]  # H, W
            elif len(images_and_masks.shape) == 2:
                return images_and_masks.shape[0], images_and_masks.shape[1]  # H, W
            else:
                raise ValueError(f"不支持的Tensor维度：{images_and_masks.shape}")

        elif isinstance(images_and_masks, np.ndarray):
            # OpenCV数组通常为 (H, W, C) 或 (H, W)
            if len(images_and_masks.shape) in (2, 3):
                return images_and_masks.shape[0], images_and_masks.shape[1]  # H, W
            else:
                raise ValueError(f"不支持的ndarray维度：{images_and_masks.shape}")

        elif isinstance(images_and_masks, Image.Image):
            # PIL Image的size为 (宽, 高)，转换为 (高, 宽)
            width, height = images_and_masks.size
            return height, width

        else:
            raise TypeError(f"不支持的数据类型：{type(images_and_masks)}，请使用PIL/ndarray/Tensor")

    @staticmethod
    def to_tensor(data: Any, data_name: str) -> torch.Tensor:
        """将数据强制转为torch.Tensor"""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Image.Image):
            return torch.from_numpy(np.array(data))
        else:
            raise TypeError(f"{data_name}类型不支持：{type(data)}，需为Tensor/ndarray/PIL.Image")

    def validate_spatial_consistency(self) -> None:
        """校验图像与掩码的空间尺寸一致性（高和宽必须相同）"""
        # 确定需要校验的索引（全部或仅第一个）
        indices = range(len(self)) if self.validate_full_dataset else [0]

        for idx in indices:
            # 提取核心数据（避免元组传入get_spatial_dims）
            image_core = self.extract_core_data(self.image_dataset[idx])
            mask_core = self.extract_core_data(self.mask_dataset[idx])

            image_height, image_width = self.get_spatial_dims(image_core)
            mask_height, mask_width = self.get_spatial_dims(mask_core)

            if (image_height, image_width) != (mask_height, mask_width):
                raise ValueError(
                    f"索引{idx}的空间尺寸不匹配：图像({image_height}, {image_width}) vs 掩码({mask_height}, {mask_width})"
                    f"\n请检查原始数据或前置的Dataset类，确保它们返回的图像和掩码尺寸一致。"
                )
        print("数据集空间尺寸校验通过。")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回（图像Tensor, 掩码Tensor）"""
        # 提取核心数据
        image_raw = self.extract_core_data(self.image_dataset[idx])
        mask_raw = self.extract_core_data(self.mask_dataset[idx])

        # 应用增强
        if self.transform_image:
            image_raw = self.transform_image(image_raw)
        if self.transform_mask:
            mask_raw = self.transform_mask(mask_raw)

        # 强制转为Tensor
        image = self.to_tensor(image_raw, "图像")
        mask = self.to_tensor(mask_raw, "掩码")

        # 掩码后处理（类型+维度）
        mask = mask.long()  # 确保为长整型（类别索引）
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)  # (1, H, W) → (H, W)
        elif mask.dim() != 2:
            print(f"警告：索引{idx}的掩码维度异常 {mask.shape}，需为(H, W)或(1, H, W)")

        return image, mask



