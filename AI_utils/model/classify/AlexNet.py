import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet_CIFAR10(nn.Module):
    """适配CIFAR-10 的 AlexNet  """

    def __init__(self, num_classes: int = 10, input_channels: int = 3) -> None:
        super().__init__()
        # 32×32 → 卷积+池化 ×3 → 4×4×256
        self.features_extract_block = nn.Sequential(
            self._conv_bn_relu(input_channels, 64),
            nn.MaxPool2d(2),                          # 32 → 16
            self._conv_bn_relu(64, 192),
            nn.MaxPool2d(2),                          # 16 → 8
            self._conv_bn_relu(192, 384),
            self._conv_bn_relu(384, 256),
            self._conv_bn_relu(256, 256),
            nn.MaxPool2d(2),                          # 8 → 4
        )
        self.classifier_block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    @staticmethod
    def _conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
        """卷积 3×3 + BN + ReLU。"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_extract_block(x)
        x = torch.flatten(x, 1)
        return self.classifier_block(x)
