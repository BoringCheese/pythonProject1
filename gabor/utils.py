import cv2
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

import torch

warnings.filterwarnings('ignore')


def rgb_yuv444(img):

    img1 = cv2.imread(img)
    blue, green, red = cv2.split(img1)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    u = -0.147 * red - 0.29 * green + 0.4359 * blue + 0.5
    v = 0.615 * red - 0.515 * green - 0.1 * blue + 0.5
    a = cv2.merge((y[:, :], u[:, :], v[:, :]))
    return a


def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg


# 定义一个低通滤波器函数，用于生成低通滤波器的频率响应
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
            eo[o * n_scales + s] = np.fft.ifft2(filter * img_fft)
    for i, img1 in enumerate(eo):
        eo[i] = np.abs(eo[i])
        eo[i] = (eo[i] - np.min(eo[i])) / (np.max(eo[i]) - np.min(eo[i]))
        eo[i] = eo[i] * 255
    return eo


def log_gabor(ig):
    n_scales = 5
    n_angles = 4
    img = cv2.imread(ig)
    H, W, _ = img.shape
    min_wavelength = 6
    multiplier = 2
    sigma_onf = 0.55
    dThetaOnSigma = 1.2
    thetaSigma = np.pi / n_angles / dThetaOnSigma

    blue, red, green = cv2.split(img)
    Y1 = 0.299 * red + 0.587 * green + 0.114 * blue
    I1 = 0.596 * red - 0.274 * green - 0.322 * blue
    Q1 = 0.211 * red - 0.523 * green + 0.312 * blue

    Y1_fft = np.fft.fft2(Y1)

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
        # dtheta = np.minimum(dtheta * n_angles * 0.5, np.pi)
        spreads[o] = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))

    #  构造集合的filter
    filter_bank = np.zeros((n_scales * n_angles, H, W))
    eo = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    for o in range(n_angles):
        for s in range(n_scales):
            filter = log_gabor_filters[s] * spreads[o]
            # ifftFilt = real(np.fft.ifft2(filter)) * sqrt(rows * cols);
            ifftFilt = np.real(np.fft.ifft2(filter)) * np.sqrt(H * W)
            filter_bank[o * n_scales + s] = ifftFilt
            eo[o * n_scales + s] = np.fft.ifft2(filter * Y1_fft)
    for i, img1 in enumerate(eo):
        eo[i] = np.abs(eo[i])
        eo[i] = (eo[i] - np.min(eo[i])) / (np.max(eo[i]) - np.min(eo[i]))
        eo[i] = eo[i] * 255

    n_temp1 = eo[0] + eo[5] + eo[10] + eo[15]
    n_temp2 = eo[1] + eo[6] + eo[11] + eo[16]
    n_temp3 = eo[2] + eo[7] + eo[12] + eo[17]
    n_temp4 = eo[3] + eo[8] + eo[13] + eo[18]
    n_temp5 = eo[4] + eo[9] + eo[14] + eo[19]
    return n_temp1


def log_gabor_3(img):
    # img = np.array(img)
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
    # a1 = np.transpose(a1, (0, 1, 2))

    n_temp_b2 = (eo_b[2] + eo_b[7] + eo_b[12] + eo_b[17]) / 4
    n_temp_r2 = (eo_r[2] + eo_r[7] + eo_r[12] + eo_r[17]) / 4
    n_temp_g2 = (eo_g[2] + eo_g[7] + eo_g[12] + eo_g[17]) / 4
    a2 = np.dstack((n_temp_r2, n_temp_g2, n_temp_b2))
    # a2 = np.transpose(a2, (0, 1, 2))

    n_temp_b3 = (eo_b[4] + eo_b[9] + eo_b[14] + eo_b[19]) / 4
    n_temp_r3 = (eo_r[4] + eo_r[9] + eo_r[14] + eo_r[19]) / 4
    n_temp_g3 = (eo_g[4] + eo_g[9] + eo_g[14] + eo_g[19]) / 4

    a3 = np.dstack((n_temp_r3, n_temp_g3, n_temp_b3))
    # a3 = np.transpose(a3, (0, 1, 2))
    a1 = torch.tensor(a1)
    a2 = torch.tensor(a2)
    a3 = torch.tensor(a3)
    return a1, a2, a3


def norm(img):
    h, w = img.shape
    mn = np.min(img)
    mx = np.max(img)
    nor = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            nor[i, j] = (img[i, j] - mn) / (mx - mn)
    return nor
