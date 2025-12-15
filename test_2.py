import torch.optim as optim
from network import WITT
from datasets import get_loader
from utils import *
torch.backends.cudnn.benchmark = True
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from datetime import datetime
import torch.nn as nn
import argparse
from distortion import *
import time
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import Resize
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# from denoise import UNet
from denoise3 import UNet2Layer
from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import sys
from thop import profile

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
            transforms.Resize((256, 256)),
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


# 载入预训练权重, 500M还挺大的 下载地址:https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
# model = SegNet(18).cuda()
# # model.load_state_dict(torch.load(r"checkpoints/vgg16_bn-6c64b313.pth"), strict=False)
# model.load_state_dict(torch.load(r"model3.pth"), strict=False)

from d2l import torch as d2l
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import requests
from torchvision.models.segmentation import deeplabv3_resnet50


# 可视化结果
def visualize_segmentation(img, seg_map, out1,out2):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title("image")

    plt.subplot(2, 2, 2)
    plt.imshow(seg_map)
    # plt.imshow(seg_map[0, :, :])
    plt.title("seg_map")


    plt.subplot(2, 2, 3)
    plt.imshow(out1)
    # plt.imshow(seg_map[0, :, :])
    plt.title("selection")


    plt.subplot(2, 2, 4)
    plt.imshow(out2)
    # plt.imshow(seg_map[0, :, :])
    plt.title("result")

    plt.show()
    plt.savefig('f2.png')
    
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.savefig('f2-1.png')

    plt.figure(figsize=(10, 10))
    plt.imshow(seg_map)
    plt.savefig('f2-2.png')

    plt.figure(figsize=(10, 10))
    plt.imshow(out1)
    plt.savefig('f2-3.png')

    plt.figure(figsize=(10, 10))
    plt.imshow(out2)
    plt.savefig('f2-4.png')



# # 将图像和分割结果可视化
# visualize_segmentation(img, output_predictions,out1,out2)

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

class ImageRestorationNet(nn.Module):
    def __init__(self):
        super(ImageRestorationNet, self).__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.encoder3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d((2, 1), stride=(2, 1))  
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(2, 1), stride=(2, 1))
        self.decoder3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU())
        
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(2, 1))
        self.decoder2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())

        self.decoder1 = nn.Conv2d(64, 3, kernel_size=1)



    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(self.pool(x))
        x = self.encoder3(self.pool(x)) 
        x = self.decoder3(self.upconv3(x))
        x = self.decoder2(self.upconv2(x))
        x = self.decoder1(x) 
        # return self.conv(x)
        return x

def calculate_psnr(img1, img2):
    img1 = 255*img1.cpu().detach().numpy()
    img2 = 255*img2.cpu().detach().numpy()
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 如果图像完全相同，PSNR为无限大
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--trainset', type=str, default='DIV2K',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['kodak', 'CLIC21'],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='WITT_W/O',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet')
# parser.add_argument('--channel-type', type=str, default='awgn',
#                     choices=['awgn', 'rayleigh'],
#                     help='wireless channel model, awgn or rayleigh')
parser.add_argument('--channel-type', type=str, default='shadowed_rice',
                    choices=['awgn', 'rayleigh','shadowed_rice'],
                    help='wireless channel model, awgn or rayleigh or shadowed_rice')
parser.add_argument('--C', type=int, default=96,
                    help='bottleneck dimension')
# parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
#                     help='random or fixed snr')
snr = 0
random_number = str(snr)
parser.add_argument('--multiple-snr', type=str, default= random_number,
                    help='random or fixed snr')
args = parser.parse_args()


class config():
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0")
    norm = False
    # logger
    print_step = 100
    plot_step = 10000
    filename = datetime.now().__str__()[:-7]
    workdir = './history/{}'.format(filename)
    log = workdir + '/Log_{}.log'.format(filename)
    samples = workdir + '/samples'
    models = workdir + '/models'
    logger = None

    # training details
    normalize = False
    learning_rate = 0.0001
    tot_epoch = 100

    if args.trainset == 'CIFAR10':
        save_model_freq = 5
        image_dims = (3, 32, 32)
        # train_data_dir = "/media/Dataset/CIFAR10/"
        train_data_dir = "/home/zwz21/下载/WITT/media/Dataset"
        # test_data_dir = "/media/Dataset/CIFAR10/"
        test_data_dir = "/home/zwz21/下载/WITT/media/Dataset"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 5
        image_dims = (3, 256, 256)
        train_data_dir = ["/home/zwz21/下载/SegNet/DLRSD/train_images"]
        if args.testset == 'kodak':
            # test_data_dir = ['/home/zwz21/下载/pythonProject/DIV2K_valid_HR']
            test_data_dir = ['/home/zwz21/下载/pythonProject/data/2K2']
        elif args.testset == 'CLIC21':
            test_data_dir = ["/home/zwz21/下载/pythonProject/DIV2K_valid_HR"]
            #test_data_dir = ['/home/zwz21/下载/pythonProject/data/2k']
        test_data_dir = ['/home/zwz21/下载/SegNet/figure']
        # test_data_dir = ['/home/zwz21/下载/SegNet/picture2_train']
        batch_size = 1
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )


if args.trainset == 'CIFAR10':
    CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).cuda()
else:
    CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).cuda()

def load_weights(model_path):
    pretrained = torch.load(model_path)
    net.load_state_dict(pretrained, strict=True)
    del pretrained


def test():
    config.isTrain = False
    net.eval()
    elapsed, psnrs, msssims, snrs, cbrs = [AverageMeter() for _ in range(5)]
    metrics = [elapsed, psnrs, msssims, snrs, cbrs]
    multiple_snr = args.multiple_snr.split(",")
    for i in range(len(multiple_snr)):
        multiple_snr[i] = int(multiple_snr[i])
    results_snr = np.zeros(len(multiple_snr))
    results_cbr = np.zeros(len(multiple_snr))
    results_psnr = np.zeros(len(multiple_snr))
    results_msssim = np.zeros(len(multiple_snr))
    for i, SNR in enumerate(multiple_snr):
        with torch.no_grad():
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    
                    image_path = "/home/zwz21/下载/SegNet/DLRSD/train_images/tenniscourt88.png"
                    # image_path = "/home/zwz21/下载/SegNet/DLRSD/train_images/buildings63.png"
                    # image_path = "/home/zwz21/下载/SegNet/DLRSD/train_images/airplane50.png"
                    img = Image.open(image_path) 
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img)
                    plt.axis('off')
                    plt.savefig('130.png')
                    plt.show()
# 定义图像预处理变换
                    preprocess = T.Compose([
                    # transforms.CenterCrop((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

# 预处理图像
                    input_tensor = preprocess(img)
                    input_batch = input_tensor.unsqueeze(0)
                    input_batch = input_batch.to(device)

# 加载预训练的DeepLabV3模型
                    model = SegNet(18)
                    model = torch.load(r"different_model7a.pth")
# model.eval()
                    model.to(device)

# 将输入图像移到计算设备上
                   
# 进行语义分割预测
                   
                    with torch.no_grad():
                         output = model(input_batch)
                         output_predictions = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
                         color_map = np.array([(166, 202, 240), (128, 128, 0 ), (0,0,128), (255, 0, 0 ), (0, 128, 0 ), (128, 0, 0 ), (255, 233, 233 ), (160, 160, 164 )
                       ,(0, 128, 128),(90, 87, 255),( 255, 255, 0 ),(255, 192, 0 ),(0, 0, 255 ),(255, 0, 192 ),(128, 0, 128 ),
                       (0, 255, 0 ),( 0, 255, 255 ),(0, 0, 0)])
                         output_predictions = color_map[output_predictions]
                         out1 = output_predictions.copy()

                    #      preprocess = T.Compose([
                    # transforms.CenterCrop((256, 256))
                    # ])
                    #      img = preprocess(img)
                         img1 = np.array(img)
                         mask1 =np.any(out1 != [128, 0, 0 ], axis=-1)
                         out1[mask1] = [255,255,255]
                         out2 = out1.copy()
                         mask2 =np.any(out1 != [255, 255, 255 ], axis=-1)
                         out2[mask2] = [0,0,0]

                         image = Image.open(image_path).convert("RGB")
                         transform1 = transforms.CenterCrop((256, 256))
                         transform = transforms.ToTensor()
                        #  img2 = transform(transform1(image))
                         img2 = transform(image)
                         out2 = img1 + out2
                         
                        #  original_tensor = torch.from_numpy(out2/255.0).float()
                        #  input_batch = original_tensor.permute(2, 0, 1).unsqueeze(0).cuda()
                        #  snr2 = torch.tensor(snr).cuda()
                        #  noise_power = torch.mean(input_batch ** 2) / (10 ** (snr2 / 10))
                        #  noise = torch.randn_like(input_batch) * torch.sqrt(noise_power)
                        #  noisy_image = input_batch + noise
                        #  restored_image = model2(noisy_image)
                        #  psnr_predictor = calculate_psnr(input_batch, restored_image)
                        #  print('predict PSNR =', psnr_predictor)

                        # #  if psnr_predictor >= 10 :

                         if snr >= -3 :
                            # if psnr_predictor >= 20 :
                            if snr >= 3:
                                # img3 = mean_blur(img2, kernel_size=3)
                                img3 = img2
                            # if psnr_predictor < 20 and psnr_predictor >= 10 :
                            if snr < 3 and snr >= -3:
                                # img3 = mean_blur(img2, kernel_size=9) 
                                img3 = img2
                            # if snr < 5 and snr >= 0:
                            #     img3 = mean_blur(img2, kernel_size=9) 
                            img3 = img3.permute(1,2,0)
                            img3 = np.array(img3)
                            # mask3 = np.any(out1 == [128, 0, 0 ], axis=-1)
                            # img3[mask3] = [0,0,0]
                            mask4 =np.any(out2 >= [255, 255, 255], axis=-1) 
                            out2[mask4] = [0,0,0]
                            # out2 = (img3*255).astype(np.int64) + out2
                            out2 = (img3*255).astype(np.int64)

                         out2[out2 > 255] = 255
                         plt.figure(figsize=(10, 10))
                        #  plt.imshow(output_predictions)
                         plt.imshow(out2)
                         plt.axis('off')
                         plt.savefig('131-2.png')
                         plt.show()
                        #  visualize_segmentation(img, output_predictions,out1,out2)

                         out2=np.array(out2).astype(np.float32)
                         out2 = out2 / 255.0
                         out2 =  torch.from_numpy(out2)
                         out2  =  out2.permute(2, 0, 1) 
                    input = torch.unsqueeze(out2, dim=0)
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)

                    # recon_image, CBR, SNR, mse, loss_G = net(input_batch, SNR)

                    input = recon_image.cpu()
                    input = torch.clamp(input, 0.0, 1.0)
                    input = torch.squeeze(input)
                    input = input.permute(1, 2, 0)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(input)
                    plt.axis('off')
                    plt.savefig('132-2.png')
                    plt.show()

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = WITT(args, config)
    # net = torch.load(r"model_2_5.pth")
    net = torch.load(r"model_no_channel_1.pth")
    net = net.cuda()

    model2 = ImageRestorationNet()
    model2 = torch.load('/home/zwz21/下载/SegNet/predictor_model_2.pth')
    model2 = model2.cuda()
    print("model2=",model2)


    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()
    test()


