import cv2
import numpy as np


def rgb_yuv444(img):

    red, green, blue = cv2.split(img)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    u = -0.147 * red - 0.29 * green + 0.4359 * blue + 0.5
    v = 0.615 * red - 0.515 * green - 0.1 * blue + 0.5
    a = cv2.merge((y[:, :], u[:, :], v[:, :]))
    return a


def yuv_rgb444(img):

    Y, U, V = cv2.split(img)
    R = Y + (V - 0.5) * 1.140
    G = Y + (U * 0.5) * -0.395 + (V - 0.5) * -0.581
    B = Y + (U - 0.5) * 2.032
    a = cv2.merge((R[:, :], G[:, :], B[:, :]))
    for i, img1 in enumerate(a):
        a[i] = (a[i] - np.min(a[i])) / (np.max(a[i]) - np.min(a[i]))
        a[i] = a[i] * 255

    return a


if __name__ == '__main__':
    img = cv2.imread("../bird.JPEG")
    a = rgb_yuv444("../bird.JPEG")
    print(a)
    print(img)
