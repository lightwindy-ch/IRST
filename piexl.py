import cv2
import numpy as np

def compare_image_similarity(image_path1, image_path2):
    # 读取两张图像
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # 检查尺寸是否一致
    if img1.shape != img2.shape:
        raise ValueError("图像尺寸不一致，无法逐像素比较。")

    # 将图像转换为灰度（可选，取决于对颜色是否敏感）
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 比较两个图像的像素是否相同
    equal_pixels = np.sum(img1 == img2)
    total_pixels = img1.size  # 总像素数

    # 计算相同像素的比例
    similarity_ratio = equal_pixels / total_pixels
    return similarity_ratio

ratio = compare_image_similarity('/home/zwz21/下载/SegNet/DLRSD/train_labels/sparseresidential42.png', '/home/zwz21/下载/SegNet/pic1.png')
print(f"图像相同像素比例: {ratio:.4f}")