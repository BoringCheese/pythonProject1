import cv2
import numpy as np
import YUV
import torch
from utils import log_gabor_3
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import time
t = tqdm(range(100))
# 模拟一些任务
for i in t:
    # 做一些耗时的操作
    time.sleep(0.1)
