import cv2
import numpy as np
import matplotlib.pyplot as plt
import utils
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')
n_scales = 5
n_angles = 4
img = cv2.imread("../bird.JPEG")
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
eo = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
# eo_r = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
# eo_g = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
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

imgg = utils.BGR_to_RGB(img)

plt.subplot(121)
plt.imshow(imgg, cmap="gray")
plt.title("original")
plt.axis("off")

plt.subplot(122)
plt.imshow(n_temp2, cmap="gray")
plt.title("iimg")
plt.axis('off')
plt.show()