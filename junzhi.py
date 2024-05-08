import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读入图像
img = cv2.imread("bird.JPEG")

# 滤波器大小（取奇数）
kernel_size = 5

# 使用均值滤波器进行滤波
filtered_img = cv2.blur(img, (kernel_size, kernel_size))

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
