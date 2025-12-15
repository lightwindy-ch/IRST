import torch
# import torch.nn as nn
import scipy.special as sc
from scipy.stats import rv_continuous
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   

# # 添加高斯白噪声
def add_noise(signal, snr):
    noise_power = torch.mean(signal ** 2) / (10 ** (snr / 10))
    noise = torch.randn_like(signal) * torch.sqrt(noise_power)
    noise.to(device)
    x = torch.from_numpy(ShadowedRiceDistribution.shadowed_rice_channel())
    x = torch.tensor(x, dtype=torch.float32).to(device) 
    print(x)                    
    h = torch.zeros(size=np.shape(signal), device=device)
    h = x+h
    return h*signal + noise


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.special as sc
from scipy.stats import rv_continuous
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ShadowedRiceDistribution(rv_continuous):
    def _pdf(self, s):
        b = 0.158
        m = 19.4
        omega = 1.29

        F1 = sc.hyp1f1(m, 1, omega * s / (2 * b * (2 * b * m + omega)))
        Ps = (2 * b * m / (2 * b * m + omega)) ** m * (1 / (2 * b)) ** (-s / (2 * b)) * F1
        return Ps

    def shadowed_rice_channel():
        np.random.seed(None)
        custom_dist = ShadowedRiceDistribution(a=0, b=1, name='custom_dist')
        samples = custom_dist.rvs(size=1)
        return samples

###
class snrEncoder(nn.Module):
    def __init__(self):
        super(snrEncoder, self).__init__()
        # self.encoder1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU())
        # self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        # self.encoder3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU())
        # self.pool = nn.MaxPool2d((2, 1), stride=(2, 1))  

        
        self.encoder1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.encoder2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)


    def forward(self, x, snr):
        # if snr >= 3:
        #     x = self.encoder1(x)
        # if snr < 3 and snr >=-3:
        #     x = self.encoder1(x)
        #     x = self.encoder2(self.pool(x))
        # if snr < -3:
        #     x = self.encoder1(x)
        #     x = self.encoder2(self.pool(x))
        #     x = self.encoder3(self.pool(x))  
        
        if snr >= 3:
            x = self.encoder1(x)
        if snr < 3 and snr >=-3:
            x = self.encoder1(x)
            x = self.encoder2(x)
        if snr < -3:
            x = self.encoder1(x)
            x = self.encoder2(x)
            x = self.encoder3(x)
        
        # x = self.encoder1(x)
        # x = self.encoder2(self.pool(x))
        # x = self.encoder3(self.pool(x))  

        # x = self.encoder1(x)
        # x = self.encoder2(x)
        # x = self.encoder3(x)
        return x


# 模拟高斯信道
class snrChannel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, signal, snr):
        # 计算信号功率
        noise_power = torch.mean(signal ** 2) / (10 ** (snr / 10))
        noise = torch.randn_like(signal) * torch.sqrt(noise_power)
        x = torch.from_numpy(ShadowedRiceDistribution.shadowed_rice_channel())
        x = torch.tensor(x, dtype=torch.float32).to(device) 
        print(x)                    
        h = torch.zeros(size=np.shape(signal), device=device)
        h = x+h
        return h*signal + noise



# 定义 U-Net 解码器
class snrDecoder(nn.Module):
    def __init__(self):
        super(snrDecoder, self).__init__()
        # self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1))
        # self.decoder3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU())
        
        # self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1))
        # self.decoder2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())

        # self.decoder1 = nn.Conv2d(64, 1, kernel_size=1)


        self.decoder3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.decoder1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        

    def forward(self, x, snr):
        # if snr >=3:
        #     x = self.decoder1(x)
        # if snr < 3 and snr >=-3:
        #     x = self.decoder2(self.upconv2(x))
        #     x = self.decoder1(x)
        # if snr < -3:
        #     x = self.decoder3(self.upconv3(x))
        #     x = self.decoder2(self.upconv2(x))
        #     x = self.decoder1(x)

        if snr >=3:
            x = self.decoder1(x)
        if snr < 3 and snr >=-3:
            x = self.decoder2(x)
            x = self.decoder1(x)
        if snr < -3:
            x = self.decoder3(x)
            x = self.decoder2(x)
            x = self.decoder1(x)

        # x = self.decoder3(self.upconv3(x))
        # x = self.decoder2(self.upconv2(x))
        # x = self.decoder1(x)

        # x = self.decoder3(x)
        # x = self.decoder2(x)
        # x = self.decoder1(x)
        return x




# # 构建模型
# snr = 14
# encoder = snrEncoder().cuda()
# channel = snrChannel().cuda()
# decoder = snrDecoder().cuda()

# # print(encoder)
# # print(decoder)
# # # 示例输入
# # input_image = torch.randn(1, 1, 64, 64)  # 假设 64x64 灰度图像

# # # 传输流程
# # encoded = encoder(input_image)
# # transmitted = channel(encoded)
# # decoded = decoder(transmitted)

# # print("输入图像尺寸:", input_image.shape)
# # print("编码后尺寸:", encoded.shape)
# # print("传输后尺寸:", transmitted.shape)
# # print("解码后尺寸:", decoded.shape)



# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

# # 生成随机数据进行训练
# num_epochs = 1000

# for epoch in range(num_epochs):
#     # 生成输入数据 (假设是 64x64 的灰度图像)
#     input_image = torch.randn(1, 256, 96).cuda()

#     # 编码
#     encoded = encoder(input_image,snr)
#     # 通过高斯信道
#     transmitted = channel(encoded,snr)
#     # 解码
#     output_image = decoder(transmitted,snr)

#     # 计算损失
#     loss = criterion(output_image, input_image)

#     # 反向传播和优化
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")