# 本项目实现了多种用于图像分类任务的神经网络架构
import torch
import torch.nn as nn
from torch.nn import init
from typing import Tuple


# 定义了各种初始化的方法
def init_weights(net, init_type='normal', gain=0.02):
    """
    通用网络参数初始化函数，支持对卷积层、线性层和批归一化层应用多种初始化策略。

    功能：
        递归遍历网络中所有模块，对符合条件的层（卷积层/线性层/批归一化层）
        根据指定的初始化类型（init_type）进行参数初始化，确保网络参数以合理分布开始训练，
        避免梯度消失/爆炸问题，加速模型收敛。

    参数：
        net (torch.nn.Module): 待初始化的网络模型（必须是nn.Module的子类，如MLP、CNN等）
        init_type (str): 初始化策略，支持以下类型：
            - 'normal': 正态分布初始化，权重~N(0, gain²)
            - 'xavier': Xavier正态分布初始化（适合tanh/sigmoid等对称激活函数）
            - 'kaiming': Kaiming正态分布初始化（适合ReLU等非对称激活函数，默认模式'fan_in'）
            - 'orthogonal': 正交初始化（减少参数间相关性，适合RNN等结构）
        gain (float): 增益参数，控制初始化分布的标准差缩放比例：
            - 对'normal'/'xavier'/'orthogonal'影响分布范围
            - 对批归一化层，控制权重的正态分布标准差（均值固定为1.0）
    使用示例：
    cnn_model = SimpleCNN()
    init_weights(cnn_model, init_type='xavier', gain=1.0)  # 适合tanh激活
    """

    def init_func(module):
        classname = module.__class__.__name__
        if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(module.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(module.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(module.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(module, 'bias') and module.bias is not None:
                init.constant_(module.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(module.weight.data, 1.0, gain)
            init.constant_(module.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


# 基于两个全连接层的适用于图像分类任务的多层感知机
class MLP2_classify(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(MLP2_classify, self).__init__()
        """
        # 参数和使用介绍
        :param input_dim: 输入特征的维度（展平后的向量长度）
        :param output_dim: 输出类别的数量（分类任务中的类别数）
        使用示例：
        model = MLP2_classify(input_dim=784, output_dim=10)
        该示例创建了一个输入维度为784（如28x28图像展平后）的MLP模型，输出10个类别（如数字分类）。
        """

        # 添加参数合法性检查
        if not isinstance(input_dim, int) or not isinstance(output_dim, int):
            raise TypeError("input_dim 和 output_dim 必须是整数")
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim 和 output_dim 必须是正整数")

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),  # pytorch创建层时参数权重默认使用Kaiming初始化，偏置为0
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        out = self.net(x)
        return out


# 基于五个全连接层的适用于图像分类任务的多层感知机
class MLP5_classify(nn.Module):
    def __init__(self, input_size: list, num_classes=None, dropout_rate=0.4):
        super(MLP5_classify, self).__init__()
        # 增加num_classes参数合法性检查
        if num_classes is None:
            raise ValueError("num_classes参数不能为空")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes必须是正整数")
        # 增加input_size参数合法性检查
        if not isinstance(input_size, list) or not all(isinstance(i, int) and i > 0 for i in input_size):
            raise ValueError("input_size必须是包含正整数的列表")
        inputs_dims = 1
        for dim in input_size:
            inputs_dims *= dim

        self.fc1 = nn.Linear(inputs_dims, out_features=1024, bias=True)
        self.BN1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, out_features=2048, bias=True)
        self.BN2 = nn.BatchNorm1d(2048)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(2048, out_features=1024, bias=True)
        self.BN3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(1024, out_features=512, bias=True)
        self.BN4 = nn.BatchNorm1d(512)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(512, out_features=num_classes, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = self.fc1(x)
        x = self.BN1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.BN2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.BN3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.BN4(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.fc5(x)
        return x


# 基于AlexNet架构的图像分类模型
class AlexNet8_classify(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.4):
        super(AlexNet8_classify, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.BN1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 32->16
        self.conv2 = nn.Conv2d(32, 128, 5, 1, 2)
        self.BN2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 16->8
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.BN3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 512, 3, 1, 1)
        self.BN4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 256, 3, 1, 1)
        self.BN5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 8
        self.fc1 = nn.Linear(256 * 4 * 4, 4096)
        self.BN1 = nn.BatchNorm1d(4096)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(4096, 1024)
        self.BN2 = nn.BatchNorm1d(1024)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.BN3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.BN4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.BN5(x)
        x = self.relu5(x)
        x = self.pool3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.BN1(x)
        x = self.relu6(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.BN2(x)
        x = self.relu7(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# 基于VGG16架构的图像分类模型
class VGG16_classify(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.4):
        super(VGG16_classify, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),  # 统一模式：Conv→BN→ReLU
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32→16
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 16→8
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 8→4
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 4→2（进一步减小特征图）
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.block6 = nn.Sequential(
            nn.Linear(512 * 2 * 2, 2048),  # 特征图2×2，维度匹配
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0), -1)
        x = self.block6(x)
        return x


class Inception(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channels2, out_channels3, out_channel4):
        super(Inception, self).__init__()
        self.route1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.route2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels2[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels2[0], out_channels2[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.route3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels3[0], kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels3[0], out_channels3[1], kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.route4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 尺寸完全不变
            nn.Conv2d(in_channel, out_channel4, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.route1(x)
        x2 = self.route2(x)
        x3 = self.route3(x)
        x4 = self.route4(x)
        return torch.cat((x1, x2, x3, x4), dim=1)  # 在通道维度合并各个支路特征，类似于模态合并


# 基于GoogLeNet架构的图像分类模型
class GoogLeNet5_classify(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.4):
        super(GoogLeNet5_classify, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 原始为7*7，步长为2卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, padding=1),  # 32->16
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # 5*5卷积替换为3*3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2,padding=1),  # 16->8，数据集尺寸太小减少部分池化
        )
        self.block3 = nn.Sequential(
            Inception(192, 64, [96, 128], [16, 32], 32),
            # 上述并行连接的输出通道数：64+128+32+32=256
            Inception(256, 128, [128, 192], [32, 96], 64),
            # 上述并行连接的输出通道数：128+192+96+64=480
            nn.BatchNorm2d(480),
            nn.MaxPool2d(3, 2, padding=1),  # 16->8
        )
        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.BatchNorm2d(832),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8->4
        )
        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            # 最后的输出通道数为1024
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d((1, 1)),  # 将输出固定为1*1大小
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class BasicBlock(nn.Module):
    """
    ResNet-18/34 的基础残差块,作为组件使用。
    每个块包含两个 3x3 的卷积层。
    """
    expansion = 1  # 通道膨胀系数，对于 BasicBlock 来说是 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.ReLU1 = nn.ReLU(inplace=True)

        self.Conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # 跳跃连接:如果维度或步长发生变化，需要通过一个 1x1 卷积来匹配 shortcut 的维度(对应原论文中的方法3)
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels),
            )
        else:
            self.shortcut = nn.Identity()  # 直接使用恒等映射
        self.ReLU2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # 保存原始输入以供跳跃连接使用
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLU1(x)
        x = self.Conv2(x)
        x = self.BN2(x)
        x += self.shortcut(identity)
        x = self.ReLU2(x)
        return x


# 适合图像分类数据集的ResNet34模型
class ResNet34_classify(nn.Module):
    def __init__(self, block, num_blocks, num_classes=None, adaptive_pooling_size: Tuple = (1, 1),first_output_channel=64):
        super(ResNet34_classify, self).__init__()
        """
        ResNet34模型，适用于图像分类任务，基于残差网络架构设计。
        参数:
            block: 残差块类型，通常为 BasicBlock。
            num_blocks: 每个阶段的残差块数量列表，例如 [3, 4, 6, 3]。
            num_classes: 分类任务的类别数。
            adaptive_pooling_size: 自适应池化层的输出尺寸，默认为 (1, 1)。
            first_output_channel: 第一层卷积的输出通道数，默认为64。
        使用示例:
            model = ResNet34(BasicBlock, [3, 4, 6, 3], num_classes=10,adaptive_pooling_size=(1,1),first_output_channel=64)
        该示例创建了一个 ResNet34 模型，适用于 10 类别的图像分类任务。
        该模型包含四个主要阶段，每个阶段由多个残差块组成，并使用自适应平均池化层将特征图尺寸调整为指定大小。
        适用于输入图像通道为1的情况，如MNIST数据集。
        """

        # 增加num_classes参数合法性检查
        if num_classes is None:
            raise ValueError("num_classes参数不能为空")
        if not isinstance(num_classes, int) or num_classes <= 0:
            raise ValueError("num_classes必须是正整数")

        self.in_channels = first_output_channel
        # 针对 MNIST 的调整:
        # 1. 输入通道为 1
        # 2. 使用 3x3, stride=1 的卷积核，替换掉原始的 7x7, stride=2
        # 3. 移除了原始的 MaxPool2d 层
        self.Conv1 = nn.Conv2d(1, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN1 = nn.BatchNorm2d(64)
        self.ReLU1 = nn.ReLU(inplace=True)

        # ResNet 的四个主要 stage
        # 只有 stage 2, 3, 4 会进行下采样 (stride=2)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)

        # 全局平均池化层和全连接分类层
        self.pool1 = nn.AdaptiveAvgPool2d(adaptive_pooling_size)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个 block 的 stride 设置为传入的 stride，其他的为 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion  # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.BN1(x)
        x = self.ReLU1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
