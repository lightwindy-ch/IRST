import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import math

# 数据集定义
class ImageDataset(Dataset):
    def __init__(self, images, snr_values):
        self.images = images
        self.snr_values = snr_values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.snr_values[idx]

# 简单的网络定义
class ImageRestorationModel(nn.Module):
    def __init__(self):
        super(ImageRestorationModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )

    def forward(self, x, snr):
        # 将SNR拼接到输入图像上
        snr = snr.view(-1, 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x = torch.cat([x, snr], dim=1)
        return self.conv(x)

# PSNR计算函数
def calculate_psnr(original, restored):
    mse = ((original - restored) ** 2).mean().item()
    psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 0 else float('inf')
    return psnr

# 训练过程
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, snr_values in dataloader:
            images, snr_values = images.cuda(), snr_values.cuda()
            optimizer.zero_grad()
            restored_images = model(images, snr_values)
            loss = criterion(restored_images, images)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

# 主程序
if __name__ == "__main__":
    # 假设输入数据是NumPy数组
    num_images = 100
    images = torch.randn((num_images, 3, 256, 256))  # 模拟原始图像
    snr_values = torch.rand((num_images,))  # 随机生成SNR值

    dataset = ImageDataset(images, snr_values)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = ImageRestorationModel().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, epochs=20)

    # 测试一个图像
    test_image = images[0:1].cuda()
    test_snr = torch.tensor(10).cuda()
    model.eval()
    with torch.no_grad():
        restored_image = model(test_image, test_snr)
        psnr = calculate_psnr(test_image, restored_image)
        print(f"PSNR: {psnr:.2f}")

