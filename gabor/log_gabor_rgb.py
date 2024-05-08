import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
n_scales = 5
n_angles = 4
img = cv2.imread("../bird.JPEG")
H, W, _ = img.shape
min_wavelength = 6
multiplier = 2
sigma_onf = 0.55
dThetaOnSigma = 1.2
thetaSigma = np.pi / n_angles / dThetaOnSigma

def lg(img):
    blue, red, green = cv2.split(img)

    blue_fft = np.fft.fft2(blue)
    red_fft = np.fft.fft2(red)
    green_fft = np.fft.fft2(green)
    blue_fft_real = blue_fft.real
    red_fft_real = red_fft.real
    green_fft_real = green_fft.real
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

    lp_filter = utils.lowpassfilter(H, W, 0.45, 15)

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
    eo_b = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_r = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_g = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_b = utils.filtering(n_angles, n_scales, eo_b, blue_fft_real, log_gabor_filters, spreads)
    eo_r = utils.filtering(n_angles, n_scales, eo_r, red_fft_real, log_gabor_filters, spreads)
    eo_g = utils.filtering(n_angles, n_scales, eo_g, green_fft_real, log_gabor_filters, spreads)

    n_temp_b = (eo_b[0] + eo_b[5] + eo_b[10] + eo_b[15]) / 4
    n_temp_r = (eo_r[0] + eo_r[5] + eo_r[10] + eo_r[15]) / 4
    n_temp_g = (eo_g[0] + eo_g[5] + eo_g[10] + eo_g[15]) / 4

    a = np.dstack((n_temp_r, n_temp_g, n_temp_b))
    a = np.transpose(a, (0, 1, 2))
    a = a.astype(int)
    return a


imgg = utils.BGR_to_RGB(img)
a = lg(img)
a1 = lg(a)
plt.subplot(121)
plt.imshow(a, cmap="gray")
plt.title("original")
plt.axis("off")

plt.subplot(122)
plt.imshow(a1)
plt.title("iimg")
plt.axis('off')
plt.show()