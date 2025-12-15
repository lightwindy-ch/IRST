import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

def calculate_psnr(image1, image2, region):
    """
    计算两张图片在指定区域的PSNR值
    :param image1: 第一张图片路径
    :param image2: 第二张图片路径
    :param region: 指定区域 (x_min, y_min, x_max, y_max)
    :return: PSNR值
    """
    img1 = np.array(Image.open(image1))
    img2 = np.array(Image.open(image2))

    # 裁剪指定区域
    x_min, y_min, x_max, y_max = region
    img1_region = img1[y_min:y_max, x_min:x_max]
    img2_region = img2[y_min:y_max, x_min:x_max]

    # 计算PSNR
    return psnr(img1_region, img2_region, data_range=img1_region.max() - img1_region.min())

# 示例使用
image1_path = '/home/zwz21/下载/SegNet/131-2 (1).png'
image2_path = '/home/zwz21/下载/SegNet/132-2 (1).png'
region = (600, 300, 700, 400)  # 指定区域(x_min, y_min, x_max, y_max)

psnr_value = calculate_psnr(image1_path, image2_path, region)

print(f"指定区域的PSNR值: {psnr_value:.2f}")

# from PIL import Image

# def crop_image(input_path, output_path, region):
#     """
#     裁剪图像指定区域并保存
#     :param input_path: 原始图片路径
#     :param output_path: 裁剪后图片保存路径
#     :param region: 指定区域 (x_min, y_min, x_max, y_max)
#     """
#     # 打开图片
#     image = Image.open(input_path)

#     # 裁剪指定区域
#     cropped = image.crop(region)

#     # 保存结果
#     cropped.save(output_path)
#     print(f"裁剪完成，保存为: {output_path}")

# # 示例使用
# image_path = "/home/zwz21/下载/SegNet/132-2 (1).png"
# cropped_path = "/home/zwz21/下载/SegNet/area/cropped3.png"
# crop_region = (200, 240, 700, 500)  # 指定区域 (左，上，右，下)

# crop_image(image_path, cropped_path, crop_region)
