# 本项目定义了一些初始化神经网络权重的方法，供模型训练时使用。
import torch.nn as nn


# 定义了各种初始化的方法
def init_weights(model, init_type='normal', gain=0.02):
    # 定义初始化方法映射字典
    init_methods = {
        'normal': lambda w: nn.init.normal_(w, 0.0, gain),
        'xavier': lambda w: nn.init.xavier_normal_(w, gain=gain),
        'kaiming': lambda w: nn.init.kaiming_normal_(w, a=0, mode='fan_in'),
        'orthogonal': lambda w: nn.init.orthogonal_(w, gain=gain)
    }

    def init_func(m):
        classname = m.__class__.__name__
        # 处理卷积层和线性层
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            if init_type not in init_methods:
                raise NotImplementedError(f'初始化方法 [{init_type}] 不在函数字典中')
            init_methods[init_type](m.weight.data)
            # 处理偏置
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        # 处理BatchNorm2d
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)
    print(f'成功使用 {init_type} 方法初始化网络')
