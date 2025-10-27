import os
import torch
from PIL import Image

def test_model(model, test_loader, device, save_dir):
    """
    :param model: 训练好的模型，用于对测试集进行预测
    :param test_loader: 测试集的DataLoader对象，提供测试数据
    :param device: 计算设备，如'cpu'或'cuda'
    :param save_dir: 预测结果保存的目录路径
    :return: 无返回值，预测结果以图像形式保存在指定目录
    使用示例：
    test_model(trained_model, test_dataloader, 'cuda', './predictions')
    该示例中，trained_model是训练好的模型，test_dataloader是测试集的DataLoader，
    计算设备为GPU，预测结果将保存在当前目录下的predictions文件夹中。
    """
    model.to(device)
    model.eval()  # 设置模型为评估模式
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for index, sample in enumerate(test_loader):
            image = sample['image'].to(device)
            image_name = 'test_image_' + str(index) + '.png'  # 假设测试集图像按顺序命名

            outputs = model(image)  # 前向传播
            predicts = (outputs > 0.5).float()  # 二值化预测结果

            # 将预测结果转换为PIL图像并保存
            pred_image = predicts.squeeze().cpu().numpy() * 255  # 转为0-255范围
            pred_image = Image.fromarray(pred_image.astype('uint8'))
            pred_image.save(os.path.join(save_dir, image_name))