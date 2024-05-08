import cv2
import numpy as np
import matplotlib.pyplot as plt
import YUV

n_scales = 5
n_angles = 4
img = cv2.imread("../bird.JPEG")

H, W, _ = img.shape
# min_wavelength：它通常选择与你所处理的图像的最显著频率特征相对应。可以基于先验知识或通过观察图像的频谱来估计。
# multiplier：它决定了尺度间的增长程度。较大的 multiplier 值会导致尺度之间的增长更快，捕获更多的频率细节，而较小的值则会导致增长较慢
# sigma_onf 是一个常数，用于调整尺度滤波器的频率响应
min_wavelength = 6
multiplier = 2
sigma_onf = 0.55

img_yuv = YUV.rgb_yuv444("../bird.JPEG")

blue, red, green = cv2.split(img)
Y1 = 0.299 * red + 0.587 * green + 0.114 * blue
I1 = 0.596 * red - 0.274 * green - 0.322 * blue
Q1 = 0.211 * red - 0.523 * green + 0.312 * blue

Y1_fft = cv2.dft(np.float32(Y1))

vi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 图像的傅里叶变换
image_fft = cv2.dft(np.float32(vi_gray))


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
# lp_filter = np.fft.ifftshift(lp_filter)

# 创建Log-Gabor滤波器集合：
log_gabor_filters = np.zeros((n_scales, H, W))

#  不同尺度
for sc in range(n_scales):
    wavelength = min_wavelength * multiplier ** sc
    log_gabor_filters[sc] = np.exp(
        (-(np.log(radius * wavelength + 1e-5)) ** 2) / (2 * np.log(sigma_onf + 1e-5) ** 2)) * lp_filter

spreads = np.zeros((n_angles, H, W))

#  创建方向滤波器集合：
for o in range(n_angles):
    angle = o * np.pi / n_angles
    ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
    dc = costheta * np.cos(angle) + sintheta * np.sin(angle)
    dtheta = abs(np.arctan2(ds, dc))
    dtheta = np.minimum(dtheta * n_angles * 0.5, np.pi)
    spreads[o] = (np.cos(dtheta) + 1) / 2

#  构造集合的filter
filter_bank = np.zeros((n_scales * n_angles, H, W))
for sc in range(n_scales):
    for o in range(n_angles):
        filter_bank[sc * n_angles + o] = log_gabor_filters[sc] * spreads[o]

# eo = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
#
# for i, filter in enumerate(filter_bank):
#     eo[i] = cv2.idft(np.multiply(filter, image_fft))

# #  可视化
# for sc in range(n_scales):
#     first = filter_bank[sc * n_angles] - np.min(filter_bank[sc * n_angles])
#     first /= np.max(first)
#     first *= 255
#     shuiping = first
#     for o in range(n_angles - 1):
#         out = filter_bank[sc * n_angles + o + 1] - np.min(filter_bank[sc * n_angles + o + 1])
#         out /= np.max(out)
#         out *= 255
#         shuiping = np.hstack((shuiping, out))
#     plt.imshow(shuiping, cmap="gray")
#     plt.show(block=True)

eo = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
for i, filter in enumerate(filter_bank):
    eo[i] = cv2.idft(np.multiply(filter, Y1_fft))
for i, img1 in enumerate(eo):
    eo[i] = (eo[i] - np.min(eo[i])) / (np.max(eo[i]) - np.min(eo[i]))
    eo[i] = eo[i] * 255

n_temp1 = eo[0] + eo[1] + eo[2] + eo[3]
n_temp2 = eo[4] + eo[5] + eo[6] + eo[7]
n_temp3 = eo[8] + eo[9] + eo[10] + eo[11]
n_temp4 = eo[12] + eo[13] + eo[14] + eo[15]
n_temp5 = eo[16] + eo[17] + eo[18] + eo[19]

imgg = BGR_to_RGB(img)

plt.subplot(121)
plt.imshow(imgg, cmap="gray")
plt.title("original")
plt.axis("off")

plt.subplot(122)
plt.imshow(n_temp4, cmap="gray")
plt.title("iimg")
plt.axis('off')
plt.show()

# for sc in range(n_scales):
#     first = eo[n_scales]
#     first /= np.max(first)
#     first *= 255
#     shuiping = first
#     for o in range(n_angles - 1):
#         out = eo[sc * n_angles + o + 1] - np.min(eo[sc * n_angles + o + 1])
#         out /= np.max(out)
#         out *= 255
#         shuiping = np.hstack((shuiping, out))
#         plt.imshow(shuiping, cmap="gray")
#
# plt.show()
