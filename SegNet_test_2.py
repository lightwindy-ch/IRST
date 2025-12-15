import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.transforms as transforms
from numpy.lib.stride_tricks import sliding_window_view


torch.manual_seed(17)
 
 
class LabelProcessor:
    cls_num = 18
    def __init__(self,file_path):
        """
        self.colormap 颜色表 [[128,128,128],[128,0,0],[],...,[]]   ['r','g','b']
        self.names 类别名
        """
        self.colormap,self.names=self.read_color_map(file_path)  
 
    def read_color_map(self,file_path):
        # 读取csv文件
        pd_read_color=pd.read_csv(file_path)
        colormap=[]
        names=[]
 
        for i in range(len(pd_read_color)):
            temp=pd_read_color.iloc[i]  # DataFrame格式的按行切片
            color=[temp['r'],temp['g'],temp['b']]
            colormap.append(color)
            names.append(temp['name'])
        return colormap,names
    
    def cm2label(self,label):
        """将RGB三通道label (h,w,3)转化为 (h,w)大小，每一个值为当前像素点的类别"""
        label = np.array(label)
        h, w, _ = label.shape
        label = label.tolist()
 
        for i in range(h):
            for j in range(w):           
                label[i][j] = self.colormap.index(label[i][j])  
        label = np.array(label,dtype='int64').reshape((h, w))
        return label
 
class CamvidDataset(Dataset):
    def __init__(self,img_dir,label_dir,file_path):
        """
        :param img_dir: 图片路径
        :param label_dir: 图片对应的label路径
        :param file_path: csv文件（colormap）路径
        """
        self.img_dir=img_dir
        self.label_dir=label_dir
 
        self.imgs=self.read_file(self.img_dir)
        self.labels=self.read_file(self.label_dir)
        
        self.label_processor=LabelProcessor(file_path)
        # 类别总数与以及类别名
        self.cls_num=self.label_processor.cls_num
        self.names=self.label_processor.names
 
    def __getitem__(self, index):
        """根据index下标索引对应的img以及label"""
        img=self.imgs[index]
        label=self.labels[index]
 
        img=Image.open(img).convert('RGB')
        label=Image.open(label).convert('RGB')
 
        img,label=self.img_transform(img,label)
 
        return img,label
 
    def __len__(self):
        if len(self.imgs)==0:
            raise Exception('Please check your img_dir'.format(self.img_dir))
        return len(self.imgs)
 
    def read_file(self,path):
        """生成每个图片路径名的列表，用于getitem中索引"""
        file_path=os.listdir(path)
        file_path_list=[os.path.join(path,img_name) for img_name in file_path]
        file_path_list.sort()
 
        return file_path_list
 
    def img_transform(self,img,label):
        """对图片做transform"""
        border_width = 12
        new_image = Image.new('RGB', (256, 256), color='black')
        new_image.paste(img)
        img = new_image
        new_image = Image.new('RGB', (256, 256), color='black')
        new_image.paste(label)
        label = new_image
        transform_img=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img=transform_img(img)
        label = self.label_processor.cm2label(label)
        label=torch.from_numpy(label)   # numpy转化为tensor
        return img,label
 
 
if __name__=='__main__':
    # 路径
    root_dir='/home/zwz21/下载/SegNet/DLRSD'
    img_path = os.path.join(root_dir,'train_images')
    label_path = os.path.join(root_dir,'train_labels')
    file_path = os.path.join(root_dir,'class_dict.csv')
    x_valid_dir = os.path.join(root_dir, 'train_images')
    y_valid_dir = os.path.join(root_dir, 'train_labels')
 
    train_data=CamvidDataset(img_path,label_path,file_path)
    val_data=CamvidDataset(x_valid_dir,y_valid_dir,file_path)
    train_loader=DataLoader(train_data,batch_size=8,shuffle=True,num_workers=0)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=True,num_workers=0)
    # for i,data in enumerate(train_loader):
    #     img_data,label_data=data
    #     print(img_data.shape,type(img_data))
    #     print(label_data.shape,type(label_data))
    #     print(label_data)

for index, (img, label) in enumerate(train_loader):
    print(img.shape)
    print(label.shape)

    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow((img[0, :, :, :].moveaxis(0, 2)))
    plt.subplot(222)
    plt.imshow(label[0, :, :])

    plt.subplot(223)
    plt.imshow((img[6, :, :, :].moveaxis(0, 2)))
    plt.subplot(224)
    plt.imshow(label[6, :, :])

    plt.show()
    # plt.savefig('d.png')

    if index == 0:
        break

# Encoder模块

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 前13层是VGG16的前13层,分为5个stage
        # 因为在下采样时要保存最大池化层的索引, 方便起见, 池化层不写在stage中
        self.stage_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.stage_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.stage_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.stage_5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, x):
        # 用来保存各层的池化索引
        pool_indices = []
        x = x.float()

        x = self.stage_1(x)
        # pool_indice_1保留了第一个池化层的索引
        x, pool_indice_1 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_1)
 
        x = self.stage_2(x)
        x, pool_indice_2 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_2)

        x = self.stage_3(x)
        x, pool_indice_3 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_3)

        x = self.stage_4(x)
        x, pool_indice_4 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_4)

        x = self.stage_5(x)
        x, pool_indice_5 = nn.MaxPool2d(2, stride=2, return_indices=True)(x)
        pool_indices.append(pool_indice_5)

        return x, pool_indices


# SegNet网络, Encoder-Decoder
class SegNet(nn.Module):
    def __init__(self, num_classes):
        super(SegNet, self).__init__()
        # 加载Encoder
        self.encoder = Encoder()
        # 上采样 从下往上, 1->2->3->4->5
        self.upsample_1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.upsample_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.upsample_3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.upsample_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.upsample_5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x, pool_indices = self.encoder(x)

        # 池化索引上采样
        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[4])
        x = self.upsample_1(x)

        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[3])
        x = self.upsample_2(x)

        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[2])
        x = self.upsample_3(x)

        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[1])
        x = self.upsample_4(x)

        x = nn.MaxUnpool2d(2, 2, padding=0)(x, pool_indices[0])
        x = self.upsample_5(x)
   
        return x

def simple_pixel_replace(img):
    # 创建结果副本并扩展处理范围
    result = np.copy(img)
    h, w, _ = img.shape
    
    # 生成三通道合并的整数表示
    rgb_packed = (img[:,:,0]<<16) | (img[:,:,1]<<8) | img[:,:,2]
    
    # 仅处理内部像素 (1,1) 到 (h-2,w-2)
    for y in range(1, h-1):
        for x in range(1, w-1):
            # 获取3x3邻域并排除中心
            neighbors = rgb_packed[y-1:y+2, x-1:x+2].flatten()
            neighbors = np.delete(neighbors, 4)  # 移除中心点
            
            # 统计各值出现次数
            unique, counts = np.unique(neighbors, return_counts=True)
            max_count = np.max(counts) if len(counts) > 0 else 0
            
            # 满足条件时执行替换
            if max_count >= 7:
                dominant_value = unique[np.argmax(counts)]
                # 解包RGB值
                result[y,x,0] = (dominant_value >> 16) & 0xFF
                result[y,x,1] = (dominant_value >> 8)  & 0xFF
                result[y,x,2] = dominant_value & 0xFF
    
    return result

def mean_blur(image_tensor, kernel_size=3):
    """
    对彩色图像（RGB）应用均值模糊
    :param image_tensor: 形状为 (C, H, W) 的张量
    :param kernel_size: 滤波核大小
    :return: 处理后的图像张量
    """
    # 创建均值滤波核
    kernel = torch.ones((3, 1, kernel_size, kernel_size)) / (kernel_size ** 2)
    
    # 对每个通道分别应用滤波
    blurred_image = F.conv2d(image_tensor.unsqueeze(0), kernel, padding=kernel_size // 2, groups=3)
    
    return blurred_image.squeeze(0)

from d2l import torch as d2l
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import requests
from torchvision.models.segmentation import deeplabv3_resnet50

# 下载示例图像
image_path = "/home/zwz21/下载/SegNet/DLRSD/train_images/tenniscourt88.png"
img = Image.open(image_path) 
# img = Image.open('/home/zwz21/下载/SegNet/DLRSD/train_images/buildings37.png')

# 定义图像预处理变换
preprocess = T.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 预处理图像
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)

# 检查是否有可用的GPU并使用它
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的DeepLabV3模型
model = SegNet(18)
model = torch.load(r"different_model7a.pth")
# model.eval()
model.to(device)

# 将输入图像移到计算设备上
input_batch = input_batch.to(device)

# 进行语义分割预测
with torch.no_grad():
    output = model(input_batch)
output_predictions = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
color_map = np.array([(166, 202, 240), (128, 128, 0 ), (0,0,128), (255, 0, 0 ), (0, 128, 0 ), (128, 0, 0 ), (255, 233, 233 ), (160, 160, 164 )
                       ,(0, 128, 128),(90, 87, 255),( 255, 255, 0 ),(255, 192, 0 ),(0, 0, 255 ),(255, 0, 192 ),(128, 0, 128 ),
                       (0, 255, 0 ),( 0, 255, 255 ),(0, 0, 0)])
output_predictions = color_map[output_predictions]
output_predictions1 = output_predictions.copy()
output_predictions1 = simple_pixel_replace(output_predictions1)
out1 = output_predictions1.copy()
# out1 = simple_pixel_replace(out1)
# print(id(out1))
img1 = np.array(img)
mask1 =np.any(out1 != [128, 0, 0 ], axis=-1)
# mask1 =np.any(out1 != [0, 0, 128 ], axis=-1)
out1[mask1] = [255,255,255]

out2 = out1.copy()
mask2 =np.any(out1 != [255, 255, 255 ], axis=-1)
out2[mask2] = [0,0,0]

# image_path = "/home/zwz21/下载/SegNet/DLRSD/train_images/tenniscourt88.png"
image = Image.open(image_path).convert("RGB")
transform = transforms.ToTensor()
img2 = transform(image)
out2 = img1 + out2

snr = 8
if snr >= 10 :
 if snr >= 20:
   img3 = mean_blur(img2, kernel_size=3)
 if snr < 20 and snr >= 15:
   img3 = mean_blur(img2, kernel_size=5) 
 if snr < 20 and snr >= 10:
   img3 = mean_blur(img2, kernel_size=9) 
 img3 = img3.permute(1,2,0)
 img3 = np.array(img3)
 mask3 = np.any(out1 == [128, 0, 0 ], axis=-1)
 img3[mask3] = [0,0,0]
# blurred_image = transforms.ToPILImage()(img3)
# blurred_image.save("215-2.png")

 mask4 =np.any(out2 >= [255, 255, 255], axis=-1) 
 out2[mask4] = [0,0,0]
 out2 = (img3*255).astype(np.int64) + out2

# print(img1)
# print(output_predictions.size)
fig = plt.figure(figsize=(256/100, 256/100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.imshow(output_predictions)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #    plt.axis('off')
    #    plt.imshow(out2)
plt.savefig('pic1.png',dpi=100,bbox_inches='tight',pad_inches=0.0)

fig = plt.figure(figsize=(256/100, 256/100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')
ax.imshow(output_predictions1)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #    plt.axis('off')
    #    plt.imshow(out2)
plt.savefig('pic2.png',dpi=100,bbox_inches='tight',pad_inches=0.0)


# 可视化结果
def visualize_segmentation(img, seg_map,seg_map1, out1,out2):
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    # plt.imshow(img)
    # plt.title("image")

    # plt.subplot(2, 2, 2)
    # plt.imshow(seg_map)
    # # plt.imshow(seg_map[0, :, :])
    # plt.title("seg_map")


    # plt.subplot(2, 2, 3)
    # plt.imshow(out1)
    # # plt.imshow(seg_map[0, :, :])
    # plt.title("selection")


    # plt.subplot(2, 2, 4)
    # plt.imshow(out2)
    # # plt.imshow(seg_map[0, :, :])
    # plt.title("result")

    # plt.show()
    # plt.savefig('f1.png')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.savefig('f3-1.png')

    plt.figure(figsize=(10, 10))
    plt.imshow(seg_map)
    plt.savefig('f3-2.png')

    plt.figure(figsize=(10, 10))
    plt.imshow(out1)
    plt.savefig('f3-3.png')

    plt.figure(figsize=(10, 10))
    plt.imshow(out2)
    plt.savefig('f3-4.png')

    plt.figure(figsize=(10, 10))
    plt.imshow(seg_map1)
    plt.savefig('f3-5.png')




# 将图像和分割结果可视化
visualize_segmentation(img, output_predictions,output_predictions1,out1,out2)

