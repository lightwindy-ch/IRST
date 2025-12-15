
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
    img_path = os.path.join(root_dir,'train_images1')
    label_path = os.path.join(root_dir,'train_labels1')
    file_path = os.path.join(root_dir,'class_dict.csv')
    x_valid_dir = os.path.join(root_dir, 'val_images')
    y_valid_dir = os.path.join(root_dir, 'val_labels')
 
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


# 载入预训练权重, 500M还挺大的 下载地址:https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
model = SegNet(18).cuda()
# model.load_state_dict(torch.load(r"checkpoints/vgg16_bn-6c64b313.pth"), strict=False)
# model.load_state_dict(torch.load(r"model5.pth"), strict=False)
# model = torch.load(r"model5a.pth")
model = torch.load(r"different_model7a.pth")

from d2l import torch as d2l

# 损失函数选用多分类交叉熵损失函数
lossf = nn.CrossEntropyLoss()
# 选用adam优化器来训练
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 训练50轮
epochs_num = 1

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    # softmax_f = nn.Softmax(dim=1)
    # soft_output = softmax_f(pred)
    # print('softmax_output:\n', soft_output.shape)
    # X = pred.cpu()
    # y = y.cpu()
    # plt.figure(figsize=(10, 10))
    # plt.subplot(221)
    # plt.imshow((X[0, :, :, :].moveaxis(0, 2)))
    # plt.subplot(222)
    # plt.imshow(y[0, :, :])
    # plt.savefig('f.png')
    
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    timer, num_batches = d2l.Timer(), len(train_iter)
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            # legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])


    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            # l, acc = d2l.train_batch_ch13(
            #     net, features, labels.long(), loss, trainer, devices)
            l, acc = train_batch_ch13(
                net, features, labels.long(), loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                # animator.add(epoch + (i + 1) / num_batches,
                #              (metric[0] / metric[2], metric[1] / metric[3],
                #               None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        # animator.add(epoch + 1, (None, None, test_acc))
        print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')
        # torch.save(net.state_dict(), '/home/zwz21/下载/SegNet/model5.pth')
        # torch.save(net, '/home/zwz21/下载/SegNet/different_model7a.pth')
    

train_ch13(model, train_loader, val_loader, lossf, optimizer, epochs_num)
