import os
import sys
import json
import pickle
import random
import numpy as np
import torch
import cv2
from torch import nn
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import warnings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


def norm(img):
    h, w = img.shape
    mn = np.min(img)
    mx = np.max(img)
    nor = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            nor[i, j] = (img[i, j] - mn) / (mx - mn)
    return nor


def lowpassfilter(H, W, cutoff, n):
    if cutoff < 0 or cutoff > 0.5:
        raise ValueError('the cutoff frequency needs to be between 0 and 0.5')

    if not n == int(n) or n < 1.0:
        raise ValueError('n must be an integer >= 1')

    xrange = np.linspace(-0.5, 0.5, W)
    yrange = np.linspace(-0.5, 0.5, H)

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    radius = np.fft.ifftshift(radius)
    return 1.0 / (1.0 + (radius / cutoff) ** (2 * n))


def filtering(n_angles, n_scales, eo, img_fft, log_gabor_filters, spreads):
    for o in range(n_angles):
        for s in range(n_scales):
            filter = log_gabor_filters[s] * spreads[o]
            eo[o * n_scales + s] = np.fft.ifft2(filter * img_fft.real)
    for i, img1 in enumerate(eo):
        eo[i] = np.abs(eo[i])
        eo[i] = (eo[i] - np.min(eo[i])) / (np.max(eo[i]) - np.min(eo[i]))
        eo[i] = eo[i] * 255
    return eo


def log_gabor1(img):
    # print(type(img))
    transf = transforms.ToTensor()
    img = transf(img)
    n_scales = 5
    n_angles = 4
    C, H, W = img.shape
    min_wavelength = 6
    multiplier = 2
    sigma_onf = 0.55
    dThetaOnSigma = 1.2
    thetaSigma = np.pi / n_angles / dThetaOnSigma

    blue, red, green = torch.split(img, split_size_or_sections=1, dim=0)
    blue = torch.squeeze(blue)
    red = torch.squeeze(red)
    green = torch.squeeze(green)
    blue_fft = np.fft.fft2(blue)
    red_fft = np.fft.fft2(red)
    green_fft = np.fft.fft2(green)

    # 定义频率和角度网格
    xrange = np.linspace(-0.5, 0.5, W)
    yrange = np.linspace(-0.5, 0.5, H)
    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)

    # numpy.fft模块中的fftshift函数可以将FFT输出中的直流分量移动到频谱的中央。ifftshift函数则是其逆操作
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    lp_filter = lowpassfilter(H, W, 0.45, 15)

    # 创建Log-Gabor滤波器集合：
    log_gabor_filters = np.zeros((n_scales, H, W))
    #  不同尺度
    for sc in range(n_scales):
        wavelength = min_wavelength * multiplier ** sc
        log_gabor_filters[sc] = np.exp(
            (-(np.log(radius * wavelength + 1e-5)) ** 2) / (2 * np.log(sigma_onf + 1e-5) ** 2)) * lp_filter

    #  创建方向滤波器集合：
    spreads = np.zeros((n_angles, H, W))
    for o in range(n_angles):
        angle = o * np.pi / n_angles
        ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
        dc = costheta * np.cos(angle) + sintheta * np.sin(angle)
        dtheta = np.abs(np.arctan2(ds, dc))
        spreads[o] = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))

    #  构造集合的filter
    filter_bank = np.zeros((n_scales * n_angles, H, W))
    eo_b = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_r = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_g = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_b = filtering(n_angles, n_scales, eo_b, blue_fft, log_gabor_filters, spreads)
    eo_r = filtering(n_angles, n_scales, eo_r, red_fft, log_gabor_filters, spreads)
    eo_g = filtering(n_angles, n_scales, eo_g, green_fft, log_gabor_filters, spreads)

    n_temp_b1 = (eo_b[0] + eo_b[5] + eo_b[10] + eo_b[15]) / 4
    n_temp_r1 = (eo_r[0] + eo_r[5] + eo_r[10] + eo_r[15]) / 4
    n_temp_g1 = (eo_g[0] + eo_g[5] + eo_g[10] + eo_g[15]) / 4
    a1 = np.dstack((n_temp_r1, n_temp_g1, n_temp_b1))
    a1 = a1.astype(int)
    a1 = a1.clip(0, 255)
    # a1 = a1*100
    # a1 = np.transpose(a1, (0, 1, 2))
    a1 = a1.astype(np.int8)
    # a1 = a1.astype(np.float32)
    # print("111")
    return a1


def log_gabor_3(img):
    img = img.cpu()
    l = img.shape[0]
    out1 = torch.empty(l, 3, 224, 224)
    out2 = torch.empty(l, 3, 224, 224)
    out3 = torch.empty(l, 3, 224, 224)
    for idx, item in enumerate(img):
        a1, a2, a3 = log_gabor1(item)
        a1 = torch.transpose(a1, 2, 0)
        a2 = torch.transpose(a2, 2, 0)
        a3 = torch.transpose(a3, 2, 0)
        out1[idx] = a1
        out2[idx] = a2
        out3[idx] = a3
    out1 = out1.to(device)
    out2 = out2.to(device)
    out3 = out3.to(device)
    # out1 = norm(out1)
    # out1 = norm(out1)
    # out1 = norm(out1)

    # out1 = out1.half()
    # out2 = out2.half()
    # out3 = out3.half()
    # out1 = out1.to(torch.float32)
    # out2 = out2.to(torch.float32)
    # out3 = out3.to(torch.float32)
    return out1, out2, out3
