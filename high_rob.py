import numpy as np
import cv2  # opencv-python
import matplotlib.pyplot as plt

img = cv2.imread("./bird.JPEG")
# 将图像从BGR格式转换为RGB格式
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 将彩色图像拆分成三个通道
img_b, img_g, img_r = cv2.split(img)

# 对每个通道进行傅里叶变换
f_b = np.fft.fft2(img_b)
f_g = np.fft.fft2(img_g)
f_r = np.fft.fft2(img_r)

# 将低频成分移到频域中心
fshift_b = np.fft.fftshift(f_b)
fshift_g = np.fft.fftshift(f_g)
fshift_r = np.fft.fftshift(f_r)

# 获取图片的行列，也就是宽高
rows, cols, c = img.shape
# 获取中心位置
crow, ccol = int(rows / 2), int(cols / 2)
# 中心低频的区域，+-30都变为0
fshift_b[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
fshift_g[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
fshift_r[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

# 将低频成分移到频域左上角
ishift_b = np.fft.ifftshift(fshift_b)
ishift_g = np.fft.ifftshift(fshift_g)
ishift_r = np.fft.ifftshift(fshift_r)

# 对每个通道进行傅里叶逆变换
iimg_b = np.abs(np.fft.ifft2(ishift_b))
iimg_g = np.abs(np.fft.ifft2(ishift_g))
iimg_r = np.abs(np.fft.ifft2(ishift_r))

# 将三个通道合并
iimg = cv2.merge([iimg_b, iimg_g, iimg_r])

# 将图像从RGB格式转换为BGR格式
# iimg = cv2.cvtColor(iimg, cv2.COLOR_RGB2BGR)

# 展示
plt.subplot(121)
plt.imshow(img)
plt.title("original")
plt.axis("off")

plt.subplot(122)
plt.imshow(iimg)
plt.title("iimg")
plt.axis('off')
plt.show()

