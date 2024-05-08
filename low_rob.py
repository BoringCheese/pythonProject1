import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("./bird.JPEG")

# BGR颜色通道分离
blue, green, red = cv2.split(img)

# 对每个颜色通道分别进行傅里叶变换
dft_blue = cv2.dft(np.float32(blue), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_green = cv2.dft(np.float32(green), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_red = cv2.dft(np.float32(red), flags=cv2.DFT_COMPLEX_OUTPUT)

# 将低频中心位置移动到频域左上角
dft_shift_blue = np.fft.fftshift(dft_blue)
dft_shift_green = np.fft.fftshift(dft_green)
dft_shift_red = np.fft.fftshift(dft_red)

# 计算中心位置，行列
rows, cols = img.shape[:2]
crow, ccol = int(rows / 2), int(cols / 2)

# 创建一个大小和图像相同的掩膜
mask = np.zeros((rows, cols, 2), np.uint8)
# 中心位置设置为1
mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# 对每个颜色通道进行频域滤波
f_shift_blue = dft_shift_blue * mask
f_shift_green = dft_shift_green * mask
f_shift_red = dft_shift_red * mask

# 将低频中心位置移回到原位置
i_shift_blue = np.fft.ifftshift(f_shift_blue)
i_shift_green = np.fft.ifftshift(f_shift_green)
i_shift_red = np.fft.ifftshift(f_shift_red)

# 对每个颜色通道进行傅里叶反变换
i_blue = cv2.idft(i_shift_blue)
i_green = cv2.idft(i_shift_green)
i_red = cv2.idft(i_shift_red)

# 合并颜色通道
filtered_img = cv2.merge((i_blue[:, :, 0], i_green[:, :, 0], i_red[:, :, 0]))

# 显示原图和滤波后的图像
plt.subplot(121)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis('off')

plt.subplot(122)
plt.imshow(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2RGB))
plt.title("Filtered")
plt.axis('off')

plt.show()
