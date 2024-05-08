import cv2
import numpy as np
import matplotlib.pyplot as plt
from jb.utils import log_gabor1
import torch
from torchvision import transforms
from PIL import Image
import warnings
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
warnings.filterwarnings('ignore')


img = Image.open("../bird.JPEG")
# n_img = np.asarray(img)
# t_img = torch.tensor(np.asarray(img))
a1 = log_gabor1(img)
a1 = (a1 - a1.min()) / (a1.max() - a1.min()) * 255
a1 = a1.astype(int)

a1 = a1.astype(np.uint8)
a1 = a1.astype(int)

plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.title("original")
plt.axis("off")

plt.subplot(122)
plt.imshow(a1)
plt.title("iimg")
plt.axis('off')
plt.show()
