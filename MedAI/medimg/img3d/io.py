"""
File: io.py
Description: 3D医学影像(DICOM序列)的输入/输出操作。
Author: zlyd-CV
License: MIT
"""
import pydicom
import numpy as np
import os


def get_dicom_data(file_path):
    """
    :param file_path: DICOM文件路径
    :return: 返回DICOM数据集对象

    使用示例：
    dicom_data = get_dicom_data("path/to/dicom/file.dcm")
    该示例中，"path/to/dicom/file.dcm"是DICOM文件的路径,函数将返回该文件对应的DICOM数据集对象。
    """
    dicom_data = pydicom.dcmread(file_path)
    return dicom_data


def get_dicom_image(file_path):
    """
    :param file_path: DICOM文件路径
    :return: 返回DICOM图像数据的二维numpy数组

    使用示例：
    dicom_image = get_dicom_image("path/to/dicom/file.dcm")
    该示例中，"path/to/dicom/file.dcm"是DICOM文件的路径,函数将返回该文件对应的图像数据。
    """

    # 读取DICOM文件
    dicom_data = pydicom.dcmread(file_path)
    # 提取像素数组
    image_array = dicom_data.pixel_array
    return image_array


def get_and_assemble_dicom_images(directory_path):
    """
    :param directory_path: 包含DICOM文件的目录路径
    :return: 返回按Z轴顺序排列的DICOM图像数据的ndarray数组

    使用示例：
    dicom_images = get_dicom_series_images("path/to/dicom/series")
    该示例中，"path/to/dicom/series"是包含DICOM文件的目录路径,函数将返回该目录下所有DICOM图像数据的ndarray数组,按Z轴顺序排列(D,H,W)
    """
    # 按dicom文件中的InstanceNumber排序
    dicom_files = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.dcm'):
            filepath = os.path.join(directory_path, filename)
            dicom_data = pydicom.dcmread(filepath)

            # 加入判断，若InstanceNumber为空直接报错
            if dicom_data.InstanceNumber is None:
                raise ValueError(
                    f"InstanceNumber is missing in DICOM file: {filepath}")

            # 获取InstanceNumber(Z轴顺序)
            instance_number = int(dicom_data.InstanceNumber)
            # 存储为元组(InstanceNumber, filepath)
            dicom_files.append((instance_number, filepath))

    # 按InstanceNumber排序
    dicom_files.sort(key=lambda x: x[0])
    # 读取排序后的DICOM图像
    # 按照排序后的文件路径(filepath)读取图像
    sorted_images = [get_dicom_image(data[1]) for data in dicom_files]
    ndarray_images = np.array(sorted_images)
    print(f"Loaded {len(sorted_images)} DICOM images from {directory_path} and back ndarray shape: {ndarray_images.shape}")
    return ndarray_images
