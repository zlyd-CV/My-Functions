"""
File: dicom.py
Description: DICOM数据处理工具(非图像数据)。
Author: zlyd-CV
License: MIT
"""


def dicom_info(ds):
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
