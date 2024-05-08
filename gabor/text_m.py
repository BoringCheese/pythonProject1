import cv2
import math
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


def BGR_to_RGB(cvimg):
    pilimg = cvimg.copy()
    pilimg[:, :, 0] = cvimg[:, :, 2]
    pilimg[:, :, 2] = cvimg[:, :, 0]
    return pilimg


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


def loggabor(img):
    n_scales = 5
    n_angles = 4

    if len(img.shape) == 2:
        rows, cols = img.shape
        Y1 = img
        I1 = np.ones(shape=(rows, cols))
        Q1 = np.ones(shape=(rows, cols))

    else:
        rows, cols, c = img.shape
        blue, red, green = cv2.split(img)
        Y1 = 0.299 * red + 0.587 * green + 0.114 * blue
        I1 = 0.596 * red - 0.274 * green - 0.322 * blue
        Q1 = 0.211 * red - 0.523 * green + 0.312 * blue
    Y1 = np.array(Y1)
    # print(Y1.shape)

    min_wavelength = 6
    multiplier = 2
    sigma_onf = 0.55
    minDimension = min(rows, cols)
    F = max(1, round(minDimension / 256))

    #    aveKernel = np.ones((F, F)) / (F * F)
    #    aveI1 = convolve(I1, aveKernel, mode='constant')
    #    I1 = aveI1[::F, ::F]
    #    aveQ1 = convolve(Q1, aveKernel, mode='constant')
    #    Q1 = aveQ1[::F, ::F]
    #    aveY1 = convolve(Y1, aveKernel, mode='constant')
    #    Y1 = aveY1[::F, ::F]

    nscale = 5
    norient = 4
    minWaveLength = 6
    mult = 2
    sigmaOnf = 0.55
    dThetaOnSigma = 1.2
    k = 2.0
    epsilon = .0001
    thetaSigma = math.pi / norient / dThetaOnSigma

    imagefft = cv2.dft(np.float32(Y1))
    # print(imagefft.shape)
    zero = np.zeros(shape=(rows, cols))
    EO = [[None] * norient for _ in range(nscale)]
    estMeanE2n = []
    logGabor = []
    ifftFilterArray = [None] * nscale
    xrange = np.linspace(-0.5, 0.5, cols)
    yrange = np.linspace(-0.5, 0.5, rows)

    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    #  不同尺度
    lp = lowpassfilter(rows, cols, 0.45, 15)
    for s in range(nscale):
        logGabor.append(0)
        wavelength = minWaveLength * mult ** s
        fo = 1.0 / wavelength
        logGabor[s] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[s] = logGabor[s] * lp
        logGabor[s][0, 0] = 0
    logGabornp = np.array(logGabor)

    #  创建方向滤波器集合：
    spread = [None] * norient
    for o in range(norient):
        angl = (o - 1) * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread[o] = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))

    #  创建方向滤波器集合:
    EnergyAll = np.zeros((rows, cols))
    AnAll = np.zeros((rows, cols))
    for o in range(norient):
        sumE_ThisOrient = zero
        sumO_ThisOrient = zero
        sumAn_ThisOrient = zero
        Energy = zero
        print("======================")
        for s in range(nscale):
            filter = logGabor[s] * spread[o]
            ifftFilt = np.real(np.fft.ifft2(filter)) * np.sqrt(rows * cols)
            ifftFilterArray[s] = ifftFilt
            print("过滤器")
            print(filter)
            filternp = np.array(filter)
            EO[s][o] = np.fft.ifft2(imagefft * filternp)

    # 可视化
    EO = np.array(EO)
    # EO1 = np.complex128(EO)
    print(type(EO))
    # EO = cv2.idft(EO)
    temp = EO[4][0] + EO[4][1] + EO[4][2] + EO[4][3]
    # temp = cv2.idft(temp)

    n_temp = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))

    n_temp = n_temp*255
    n_temp = np.abs(n_temp)
    # n_temp = np.round(n_temp)
    iimg = temp
    #img = BGR_to_RGB(img)
    plt.subplot(121)
    plt.imshow(img, cmap="gray")
    plt.title("original")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(n_temp, cmap="gray")
    plt.title("iimg")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    img1 = cv2.imread("../bird.JPEG", 0)
    loggabor(img1)
