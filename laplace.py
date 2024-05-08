import cv2
import numpy as np

# 读入图像
img = cv2.imread('bird.JPEG')

# 定义拉普拉斯核
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])

# 对图像进行卷积
filtered_img = cv2.filter2D(img, -1, kernel)

# 显示原图和滤波后的图像
cv2.imshow('Original', img)
cv2.imshow('Filtered', filtered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
