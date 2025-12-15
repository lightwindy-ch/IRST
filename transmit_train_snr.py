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
# from snr_coding import SNRCoding
from snr_coding import snrEncoder
from snr_coding import snrDecoder
from snr_coding import snrChannel
# from snr_coding import add_noise
# from channel import Channel
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from decoder import WITT_Decoder


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
from datasets import HR_image

torch.manual_seed(17)


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
# snr = random.randint(-10, -4)
snr = random.randint(-10, 10)
random_number = str(snr)
parser.add_argument('--multiple-snr', type=str, default= random_number,
                    help='random or fixed snr')

# parser.add_argument('--multiple-snr', type=str, default='20',
#                     help='random or fixed snr')
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
        train_data_dir = ["/home/zwz21/下载/SegNet/DLRSD/train_images1"]
        # train_data_dir = ["/home/zwz21/下载/SegNet/picture3_train"]
        if args.testset == 'kodak':
            # test_data_dir = ['/home/zwz21/下载/pythonProject/DIV2K_valid_HR']
            test_data_dir = ['/home/zwz21/下载/pythonProject/data/2K2']
        elif args.testset == 'CLIC21':
            test_data_dir = ["/home/zwz21/下载/pythonProject/DIV2K_valid_HR"]
            #test_data_dir = ['/home/zwz21/下载/pythonProject/data/2k']
        test_data_dir = ['/home/zwz21/下载/SegNet/DLRSD/train_images']
        # test_data_dir = ['/home/zwz21/下载/SegNet/DLRSD/p1']
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


def train_one_epoch(args):
    net.train()
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    global global_step
    if args.trainset == 'CIFAR10':
        for batch_idx, (input, label) in enumerate(train_loader):
            start_time = time.time()
        
    else:
        for batch_idx, input in enumerate(train_loader):
            start_time = time.time()
            global_step += 1
            input = input.to(device)

            feature = net(input)
    
            # 编码
            encoded = encoder(feature,snr)

            # 通过高斯信道
            transmitted = channel(encoded,snr)

            # 解码
            noisy_feature = decoder(transmitted,snr)
            
            # noisy_feature = add_noise(feature, snr) 
            loss2 = criterion2(noisy_feature, feature)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            
            # for name, param in model.named_parameters():
            #       if param.requires_grad:
            #            print(f"{name} 的梯度:\n{param.grad}")

            # feature, chan_param, demodel, decoder2 = net(input)
            # encoded = encoder(feature,snr)
            # # 通过高斯信道
            # transmitted = channel(encoded,snr)
            # # 解码
            # noisy_feature = decoder(transmitted,snr)            
            # # noisy_feature = add_noise(feature, snr)             
            # output = decoder2(feature, chan_param, demodel)
            # loss2 = criterion2(output, input)
            # optimizer2.zero_grad()
            # loss2.backward()
            # optimizer2.step()
        print(f'Loss: {loss2.item():.4f}')

            
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
            if args.trainset == 'CIFAR10':
                for batch_idx, (input, label) in enumerate(test_loader):
                    start_time = time.time()
 
            else:
                for batch_idx, input in enumerate(test_loader):
                    start_time = time.time()
                    input = input.cuda()
                    recon_image, CBR, SNR, mse, loss_G = net(input, SNR)
                    elapsed.update(time.time() - start_time)
                    cbrs.update(CBR)
                    snrs.update(SNR)
                    if mse.item() > 0:
                        psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                        psnrs.update(psnr.item())
                        msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                        msssims.update(msssim)
                    else:
                        psnrs.update(100)
                        msssims.update(100)

                    log = (' | '.join([
                        f'Time {elapsed.val:.3f}',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f}',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {cur_lr}',
                    ]))
                    logger.info(log)
        results_snr[i] = snrs.avg
        results_cbr[i] = cbrs.avg
        results_psnr[i] = psnrs.avg
        results_msssim[i] = msssims.avg
        for t in metrics:
            t.clear()

    print("SNR: {}" .format(results_snr.tolist()))
    print("CBR: {}".format(results_cbr.tolist()))
    print("PSNR: {}" .format(results_psnr.tolist()))
    print("MS-SSIM: {}".format(results_msssim.tolist()))
    print("Finish Test!")


class MSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, input, target):
        tensor = target
          # 创建掩码，找到所有不等于 [1,1,1] 的位置
        mask = torch.any(tensor != torch.tensor([1, 1, 1]).view(1, 3, 1, 1).cuda(), dim=1, keepdim=True)
        # 使用 torch.where 处理掩码
        tensor = torch.where(mask, tensor * 3, tensor)
        input = torch.where(mask, input * 3, input)
        squared_diff = (input - target) ** 2
        # squared_diff = (input - tensor) ** 2
        # 根据reduction参数返回不同结果
        if self.reduction == 'mean':
            return torch.mean(squared_diff)
        elif self.reduction == 'sum':
            return torch.sum(squared_diff)
        else:  # 'none'
            return squared_diff

if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = WITT(args, config)
    net = net.cuda()
    net = torch.load(r"model_no_channel.pth")
     
    encoder = snrEncoder().cuda()

    # encoder = torch.load(r"snrencoder_-10_model2_1.pth")
    # for param in encoder.encoder1.parameters():
    #     param.requires_grad = False
    # for param in encoder.encoder2.parameters():
    #     param.requires_grad = False
    # for param in encoder.encoder3.parameters():
    #     param.requires_grad = False
    
    
    channel = snrChannel().cuda()

    decoder = snrDecoder().cuda()

    # decoder= torch.load(r"snrdecoder_-10_model2_1.pth")
    # for param in decoder.decoder1.parameters():
    #     param.requires_grad = False
    # # for param in decoder.upconv2.parameters():
    # #     param.requires_grad = False
    # for param in decoder.decoder2.parameters():
    #     param.requires_grad = False
    # for param in decoder.decoder3.parameters():
    #     param.requires_grad = False
  
    criterion2 = nn.MSELoss()
    # criterion2 = MSELoss()
    optimizer2 = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)

    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    train_loader, test_loader = get_loader(args, config)
    cur_lr = config.learning_rate
    optimizer = optim.Adam(model_params, lr=cur_lr)
    global_step = 0
    steps_epoch = global_step // train_loader.__len__()

    # if args.training:
    #     for epoch in range(steps_epoch, config.tot_epoch):
    #         train_one_epoch(args)
    #         if (epoch + 1) % config.save_model_freq == 0:
    #             save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
    #             test()
    # else:
    #     test()

    # test()

    for epoch in range(steps_epoch, config.tot_epoch):
        train_one_epoch(args)
    #     # if (epoch + 1) % config.save_model_freq == 0:
    #         # save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
    #         # torch.save(net, '/home/zwz21/下载/SegNet/model_2_5.pth')
    #         # test()
    # test()
        print("Epoch: ",epoch)
        if (epoch + 1) % config.save_model_freq == 0:
            # save_model(net, save_path=config.models + '/{}_EP{}.model'.format(config.filename, epoch + 1))
            # torch.save(net2, '/home/zwz21/下载/SegNet/snr_15_20_model_4.pth')

            # torch.save(encoder, '/home/zwz21/下载/SegNet/snrencoder_-10_model2_1.pth')
            # torch.save(decoder, '/home/zwz21/下载/SegNet/snrdecoder_-10_model2_1.pth')

            torch.save(encoder, '/home/zwz21/下载/SegNet/snrencoder_all_model_1.pth')
            torch.save(decoder, '/home/zwz21/下载/SegNet/snrdecoder_all_model_1.pth')
            # test()
       



