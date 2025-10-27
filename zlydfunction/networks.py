import torch.nn as nn
from torch.nn import init


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


# 基于两个全连接层的多层感知机
class MLP(nn.Module):
    """
    多层感知机（MLP）模型，由两个全连接层和ReLU激活函数组成，适用于图像分类等任务。

    模型结构：
    输入 → 展平层（Flatten） → 全连接层（input_dim → 256） → ReLU激活 → 全连接层（256 → output_dim）

    适用场景：
    - 处理展平后的高维特征（如图像数据展平后的向量）
    - 分类任务（输出维度对应类别数）
    """
    def __init__(self,input_dim:int,output_dim:int):
        super(MLP, self).__init__()
        """
        :param input_dim:输入的维度，例如CIFAR-10的输入维度应该是3*32*32=3072
        :param output_dim:输出的维度，CIFAR-10的输出维度应该是10
        维度：分为数据结构维度和数学维度，前者表示数组的括号嵌套数，后者表示成员的个数
        
        使用示例：model = MLP(input_dim=3072, output_dim=10)
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
