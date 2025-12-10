"""
File: dicom.py
Description: DICOM数据处理工具(非图像数据)。
Author: zlyd-CV
License: MIT
"""
import csv
import os
import pydicom


def dicom_info(ds, write_csv=None):
    """获取并打印DICOM关键影像信息(支持CT/MRI/PET),不反悔"""
    m = getattr(ds, 'Modality', 'Unknown')

    # 基础Tag
    tags = [
        ('Modality', '模态'), ('SeriesDescription', '描述'),
        ('Rows', '高'), ('Columns', '宽'), ('PixelSpacing', '像素间距'),
        ('SliceThickness', '层厚'), ('SpacingBetweenSlices', '层间距'),
        ('BitsStored', '位深'), ('PhotometricInterpretation', '光度')
    ]

    # 模态特定Tag
    if m == 'CT':
        tags.extend([('RescaleIntercept', '截距(HU)'), ('RescaleSlope', '斜率(HU)'),
                     ('WindowCenter', '窗位'), ('WindowWidth', '窗宽')])
    elif m == 'MR':
        tags.extend([('RepetitionTime', 'TR'), ('EchoTime', 'TE'), ('InversionTime', 'TI'),
                     ('FlipAngle', '翻转角'), ('MagneticFieldStrength', '磁场强度')])
    elif m == 'PT':
        tags.extend([('Units', '单位'), ('RescaleIntercept', '截距'), ('RescaleSlope', '斜率'),
                     ('PatientWeight', '体重'), ('DecayCorrection', '衰减校正')])

    print(f"=== DICOM Info ({m}) ===")
    for k, label in tags:
        val = getattr(ds, k, None)  # 获取属性值
        if val is not None:
            print(f"{label:<10} ({k}): {val}")

    # PET特殊处理: 放射性药物信息(用于计算SUV)
    if m == 'PT' and hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
        try:
            seq = ds.RadiopharmaceuticalInformationSequence[0]
            print(
                f"{'总剂量':<10} (RadionuclideTotalDose): {getattr(seq, 'RadionuclideTotalDose', 'N/A')}")
            print(
                f"{'注射时间':<10} (RadiopharmaceuticalStartTime): {getattr(seq, 'RadiopharmaceuticalStartTime', 'N/A')}")
        except:
            pass

    if write_csv:
        # 增加类型检查,确保write_csv为布尔值或字符串,布尔值True时使用默认文件名
        if isinstance(write_csv, bool):
            write_csv = 'dicom_info.csv'
        if not isinstance(write_csv, str):
            return

        headers = ['Modality', 'SeriesDescription', 'Rows', 'Columns', 'PixelSpacing', 'SliceThickness',
                   'SpacingBetweenSlices', 'BitsStored', 'PhotometricInterpretation', 'RescaleIntercept',
                   'RescaleSlope', 'WindowCenter', 'WindowWidth', 'RepetitionTime', 'EchoTime', 'InversionTime',
                   'FlipAngle', 'MagneticFieldStrength', 'Units', 'PatientWeight', 'DecayCorrection',
                   'RadionuclideTotalDose', 'RadiopharmaceuticalStartTime']
        row = {k: getattr(ds, k, '\\') for k in headers}
        for k in row:
            if row[k] is None:
                row[k] = '\\'

        if m == 'PT' and hasattr(ds, 'RadiopharmaceuticalInformationSequence'):
            try:
                seq = ds.RadiopharmaceuticalInformationSequence[0]
                row['RadionuclideTotalDose'] = getattr(
                    seq, 'RadionuclideTotalDose', '\\')
                row['RadiopharmaceuticalStartTime'] = getattr(
                    seq, 'RadiopharmaceuticalStartTime', '\\')
            except:
                pass

        is_exist = os.path.exists(write_csv)
        with open(write_csv, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not is_exist:
                writer.writeheader()
            writer.writerow(row)


def load_dicoms(root_dir):
    """递归加载目录下所有DICOM文件(鲁棒模式)"""
    dicom_list = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            try:
                # force=True尝试强制读取,增加鲁棒性
                dicom_list.append(pydicom.dcmread(os.path.join(root, f), force=True))
            except:
                pass
    return dicom_list


def iter_dicoms(root_dir):
    """递归加载目录下所有DICOM文件(生成器模式,省内存)"""
    for root, _, files in os.walk(root_dir):
        for f in files:
            try:
                # force=True尝试强制读取,增加鲁棒性,返回pydicom Dataset对象的生成器,避免内存占用过大
                yield pydicom.dcmread(os.path.join(root, f), force=True)
            except:
                pass


def dicom_list_info(ds_list, write_csv=None):
    """批量查看DICOM信息"""
    if isinstance(write_csv, bool) and write_csv:
        write_csv = 'dicom_info.csv'
    last_dir = None  # 初始化状态变量：用于记录"上一次"处理的文件是在哪个文件夹
    for ds in ds_list:
        curr_dir = os.path.dirname(ds.filename) if hasattr(
            ds, 'filename') else None

        if last_dir and curr_dir != last_dir:  # 如果当前文件夹与上一次不同，打印空行分隔
            print()
            if isinstance(write_csv, str):
                with open(write_csv, 'a') as f:
                    f.write('\n')

        dicom_info(ds, write_csv)
        last_dir = curr_dir
