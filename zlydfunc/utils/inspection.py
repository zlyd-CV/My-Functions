"""
File: inspection.py
Description: 项目检查工具（如生成目录结构）。
Author: zlyd-CV
License: MIT
"""
# 本项目用于一些检查功能的实现
import os


# 生成目录结构
def generate_tree(path, indent=""):
    """
    :param path:遍历路径的根目录，一般为"."
    :param indent:不需要传参,用于递归时控制缩进
    :return:无返回值,在控制台打印目录结构

    使用示例：
    generate_tree(".")
    """
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        print(f"{indent}├── {item}")
        if os.path.isdir(full_path):
            generate_tree(full_path, indent + "│   ")


# 检查软件包依赖
def check_packages(required_packages=None):
    """
    检查并打印缺失的依赖包
    使用示例:
    check_packages()
    """
    if required_packages is None:
        required_packages = ['numpy', 'torch',
                             'PIL', 'pydicom', 'SimpleITK', 'monai']

    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print("缺失以下软件包，请安装：")
        for i, pkg in enumerate(missing, 1):
            print(f"{i}: {pkg}")
    else:
        print(f"所有依赖包均已安装: {required_packages}")


if __name__ == "__main__":
    generate_tree("..")  # 从当前目录开始生成目录结构
    check_packages()  # 检查依赖包
