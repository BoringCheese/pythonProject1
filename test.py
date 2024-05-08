import cv2
import math
import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


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

    min_wavelength = 6
    multiplier = 2
    sigma_onf = 0.55
    minDimension = min(rows, cols)
    F = max(1, round(minDimension / 256))

    aveKernel = np.ones((F, F)) / (F * F)
    aveI1 = convolve(I1, aveKernel, mode='constant')
    I1 = aveI1[::F, ::F]
    aveQ1 = convolve(Q1, aveKernel, mode='constant')
    Q1 = aveQ1[::F, ::F]
    aveY1 = convolve(Y1, aveKernel, mode='constant')
    Y1 = aveY1[::F, ::F]

    nscale = 5
    norient = 4
    minWaveLength = 6
    mult = 2
    sigmaOnf = 0.55
    dThetaOnSigma = 1.2
    k = 2.0
    epsilon = .0001
    thetaSigma = math.pi / norient / dThetaOnSigma

    imagefft = np.fft.fft2(Y1)
    zero = np.zeros(shape=(rows, cols))
    EO = [[None] * norient for _ in range(nscale)]
    estMeanE2n = []
    logGabor = []
    ifftFilterArray = [None] * nscale
    if cols % 2:
        xrange = np.arange(-((cols - 1 / 2) / (cols - 1)), ((cols - 1) / 2) / (cols - 1))
    else:
        xrange = np.arange(-((cols / 2) / cols), (cols / 2 - 1) / cols)

    if rows % 2:
        yrange = np.arange(-((rows - 1 / 2) / (rows - 1)), ((rows - 1) / 2) / (rows - 1))
    else:
        yrange = np.arange(-((rows / 2) / rows), (rows / 2 - 1) / rows)

    [x, y] = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    lp = lowpassfilter(rows, cols, 0.45, 15)
    '''for s in range(nscale):
        logGabor.append(0)
        wavelength = minWaveLength * mult ** (s - 1)
        fo = 1.0 / wavelength
        logGabor[s] = np.exp((-(np.log(radius / fo)) ** 2) / (2 * np.log(sigmaOnf) ** 2))
        logGabor[s] = logGabor[s] * lp
        logGabor[s][0, 0] = 0'''

    # 创建Log-Gabor滤波器集合：
    log_gabor_filters = np.zeros((n_scales, rows, cols))
    for sc in range(n_scales):
        wavelength = min_wavelength * multiplier ** sc
        log_gabor_filters[sc] = np.exp(
            (-(np.log(radius * wavelength + 1e-5)) ** 2) / (2 * np.log(sigma_onf + 1e-5) ** 2)) * lp

    '''    spread = [None] * norient
    for o in range(norient):
        angl = (o - 1) * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread[o] = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))
        '''
    spreads = np.zeros((n_angles, rows, cols))
    for o in range(n_angles):
        angle = o * np.pi / n_angles
        ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
        dc = costheta * np.cos(angle) + sintheta * np.sin(angle)
        dtheta = abs(np.arctan2(ds, dc))
        dtheta = np.minimum(dtheta * n_angles * 0.5, np.pi)
        spreads[o] = (np.cos(dtheta) + 1) / 2
    print(spreads.shape)
    EnergyAll = np.zeros((rows, cols))
    AnAll = np.zeros((rows, cols))


if __name__ == '__main__':
    img1 = cv2.imread("./bird.JPEG")
    loggabor(img1)
