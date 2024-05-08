import numpy as np
import cv2  # opencv-python
import matplotlib.pyplot as plt


def highf():
    img = cv2.imread("./bird.JPEG", 0)
    # 傅里叶变换
    f = np.fft.fft2(img)
    # 得到左上角低频成分放到中间
    fshift = np.fft.fftshift(f)

    # 得到图片的行列，也就是宽高
    rows, cols = img.shape
    # 获取中心位置
    crow, ccol = int(rows / 2), int(cols / 2)
    # 中心低频的区域，+-30都变为0
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    # 低频中心位置移动到左上角
    ishift = np.fft.ifftshift(fshift)
    # 傅里叶逆变换
    iomage = np.fft.ifft2(ishift)
    # 取绝对值
    iimg = np.abs(iomage)

    # 展示
    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.title("original")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(iimg, cmap="gray")
    plt.title("iimg")
    plt.axis('off')
    plt.show()


def lowf():
    img = cv2.imread("./bird.JPEG", 0)
    # 傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 低频左上角移动到中心位置
    dftShift = np.fft.fftshift(dft)

    # 计算中心位置，行列
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    # 眼膜图像，2通道
    mask = np.zeros((rows, cols, 2), np.uint8)
    # 中心位置设置成1
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

    # 眼膜和频谱图像进行乘法，得到低通滤波
    fShift = dftShift * mask
    # 低频移动到左上角
    ishift = np.fft.ifftshift(fShift)

    # 傅里叶反变换
    iImg = cv2.idft(ishift)
    # 将俩个通道的图像转换为灰度图像
    Imgre = cv2.magnitude(iImg[:, :, 0], iImg[:, :, 1])

    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.title("original")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(Imgre, cmap='gray')
    plt.title('inverse')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    lowf()
