# 本项目定义了用于数据集划分的类和方法，部分功能可能和ReadDatasets.py中重复，但为了模块化和职责单一，仍然单独实现
from typing import Optional, Tuple
import torch
from torch.utils.data import random_split, Dataset

# 实现随机拆分交叉验证
class Shuffle_Split_Cross_Validation:
    """
    该类实现了数据集划分中的随机交叉验证，在机器学习中将数据集先划分成测试集独立于训练集和验证集存在，这个测试集将不再作为训练数据，
    而是在训练与验证结束后评估模型的泛化能力。使用该类方法时应在每个epoch循环内使用，确保划分的测试集固定且训练集和验证集在每次epoch中不同。

    核心功能：
    从原始数据集中分割出训练集，验证集和测试集

    适应场景：
    数据集采用随机交叉验证训练模型的场景

    使用样例：
    spliter=Shuffle_Split_Cross_Validation()
    train_and_validation_data,test_data = spliter.split_dataset_train_and_test(total_data,default_validate_rate = 0.2,
                 default_fixed_seed = 42)
    train_data, validate_data = spliter.split_dataset_train_and_validation(train_and_validation_data, validate_rate=validate_rate)
    """

    def __init__(self, default_test_rate: float = 0.2, default_validate_rate: float = 0.2,
                 default_fixed_seed: int = 42):
        """
        :param default_test_rate:默认测试集划分比例，为0.2
        :param default_validate_rate:默认验证集划分比例，为0.2
        :param default_fixed_seed:默认随机数生成种子，用于固定测试集的划分，默认为42
        X:Type是变量/参数的类型注解，表示预期数据类型，但即使用户指定的不是该数据类型也不会报错
        """
        # 判断输入是否合法
        if not (0 < default_test_rate < 1 and 0 < default_validate_rate < 1):
            raise ValueError("default_test_rate and default_validate_rate must be between 0 and 1.")

        # 初始化属性
        self.default_test_rate = default_test_rate
        self.default_validate_rate = default_validate_rate
        self.default_fixed_seed = default_fixed_seed

    # 将原始数据集划分出(训练集,验证集)和(测试集)
    def split_dataset_train_and_test(self, dataset: Dataset, test_rate: Optional[float] = None,
                                     seed: Optional[int] = None) -> Tuple[Dataset, Dataset]:
        """
        :param dataset: dataset对象，完整的数据集
        :param test_rate: 测试集所占的比例
        :param seed: 随机种子，默认42，因为要保证每次执行代码划分分出的训练集和测试集一致，所以固定种子
        :return: 返回两个dataset对象：(训练集和验证集集合)，测试集
        Optional[X]表示数据类型既可以是X也可以是None
        """
        # 合法性检查
        test_rate = test_rate if test_rate is not None else self.default_test_rate
        seed = seed if seed is not None else self.default_fixed_seed
        if not (0 < test_rate < 1):
            raise ValueError("test_rate must be between 0 and 1")

        # 自定义Dataset实现了len方法后才可被len()处理
        total_size = len(dataset)
        test_size = int(test_rate * total_size)
        train_and_validation_size = total_size - test_size
        generator = torch.Generator().manual_seed(seed)
        # 固定种子划分，保证每个epoch划分出结果一致
        train_and_validation_dataset, test_dataset = random_split(dataset, [train_and_validation_size, test_size],
                                                                  generator=generator)
        return train_and_validation_dataset, test_dataset

    # 将(训练集,验证集)划分出(训练集)和(验证集)
    def split_dataset_train_and_validation(self, dataset: Dataset, validate_rate: Optional[float] = None) \
            -> Tuple[Dataset, Dataset]:
        """
        :param dataset: Dataset数据集，正常来说是包含了(训练集,验证集)的数据集部分
        :param validate_rate: 验证集划分比例，推荐0.3~0.5，确保模型每次训练都有大量不同样本避免过拟合
        :return: 返回训练集和验证集的dataset对象
        """
        # 合法性检查
        validate_rate = validate_rate if validate_rate is not None else self.default_validate_rate
        seed = torch.randint(0, 2 ** 32, (1,)).item()  # 使用torch生成随机种子
        if not (0 < validate_rate < 1):
            raise ValueError("validate_rate must be between 0 and 1")

        total_size = len(dataset)
        validate_size = int(validate_rate * total_size)
        train_size = total_size - validate_size
        generator = torch.Generator().manual_seed(seed)
        # 随机种子划分，确保每个epoch划分结果不一致
        train_dataset, validate_dataset = random_split(
            dataset, [train_size, validate_size], generator=generator)
        return train_dataset, validate_dataset
