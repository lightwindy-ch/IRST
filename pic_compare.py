from PIL import Image
import numpy as np

# def compare_images(img1_path, img2_path):
#     img1 = Image.open(img1_path)
#     img2 = Image.open(img2_path)
#     if img1.size != img2.size:
#         return False
#     img1_np = np.array(img1)
#     img2_np = np.array(img2)
#     return np.array_equal(img1_np, img2_np)

# print(compare_images('/home/zwz21/下载/SegNet/pic1.png', '/home/zwz21/下载/SegNet/pic2.png'))

from skimage.metrics import structural_similarity as ssim
import cv2

# 读取两个PNG图像
# image1 = cv2.imread('/home/zwz21/下载/SegNet/131-2 (1).png', cv2.IMREAD_GRAYSCALE)
# image1 = cv2.imread('/home/zwz21/下载/SegNet/130.png', cv2.IMREAD_GRAYSCALE)
# image1 = cv2.imread('/home/zwz21/下载/SegNet/jscc.png', cv2.IMREAD_GRAYSCALE)
# image1 = cv2.imread('/home/zwz21/下载/SegNet/ddpm2.png', cv2.IMREAD_GRAYSCALE)
# image1 = cv2.imread('/home/zwz21/下载/SegNet/DLRSD/val_labels/sparseresidential92.png', cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread('/home/zwz21/下载/SegNet/area/cropped2.png', cv2.IMREAD_GRAYSCALE)

# image2 = cv2.imread('/home/zwz21/下载/SegNet/132-2 (1).png', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('/home/zwz21/下载/SegNet/jscc-1.png', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('/home/zwz21/下载/SegNet/ddpm2-2.png', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('/home/zwz21/下载/SegNet/pic1.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('/home/zwz21/下载/SegNet/area/cropped1.png', cv2.IMREAD_GRAYSCALE)

# image3 = cv2.imread('/home/zwz21/下载/SegNet/132-1 (1).png', cv2.IMREAD_GRAYSCALE)
# image3 = cv2.imread('/home/zwz21/下载/SegNet/pic2.png', cv2.IMREAD_GRAYSCALE)
# image3 = cv2.imread('/home/zwz21/下载/SegNet/DLRSD/train_labels/tenniscourt22.png', cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread('/home/zwz21/下载/SegNet/DLRSD/train_labels/buildings37.png', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('/home/zwz21/下载/SegNet/area/cropped3.png', cv2.IMREAD_GRAYSCALE)

# 计算SSIM
ssim_value, _ = ssim(image1, image3, full=True)
# ssim_value1, _ = ssim(image2, image3, full=True)
ssim_value2, _ = ssim(image1, image2, full=True)

print("SSIM值:",ssim_value)
print("SSIM值:", ssim_value2)


# from skimage import io
# from skimage.metrics import structural_similarity as ssim
# import numpy as np
 
# # 读取两个PNG图像
# image1 = io.imread('/home/zwz21/下载/SegNet/pic1.png')
# image2 = io.imread('/home/zwz21/下载/SegNet/pic2.png')
 
# # 确保两个图像具有相同的尺寸
# if image1.shape != image2.shape:
#     raise ValueError("Images must have the same dimensions")
 
# # 计算SSIM值
# ssim_index, _ = ssim(image1, image2, full=True)
# print(f"SSIM Index: {ssim_index}")

import cv2
import numpy as np

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # 如果图像完全相同，PSNR为无限大
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# 读取两个PNG图像
# image1 = cv2.imread('image1.png')
# image2 = cv2.imread('image2.png')

# 计算PSNR
psnr_value = calculate_psnr(image1, image3)
# psnr_value1 = calculate_psnr(image2, image3)
psnr_value2 = calculate_psnr(image1, image2)

print("图像间的PSNR值:",psnr_value) 
print("图像间的PSNR值:",psnr_value2) 