import cv2
import numpy as np


def laplace(img):
    # 读入图像
    img = cv2.imread('bird.JPEG')
    # 定义拉普拉斯核
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # 对图像进行卷积
    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img


def mean_filter(img):
    # 读入图像
    img = cv2.imread("bird.JPEG")
    # 滤波器大小（取奇数）
    kernel_size = 5
    # 使用均值滤波器进行滤波
    filtered_img = cv2.blur(img, (kernel_size, kernel_size))
    return filtered_img
