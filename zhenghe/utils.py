import os
import sys
import json
import pickle
import random
import numpy as np
import torch
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

import warnings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    la = torch.empty(0)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        la = torch.cat((la, labels), dim=0)
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def norm(img):
    h, w = img.shape
    mn = np.min(img)
    mx = np.max(img)
    nor = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            nor[i, j] = (img[i, j] - mn) / (mx - mn)
    return nor


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


def filtering(n_angles, n_scales, eo, img_fft, log_gabor_filters, spreads):
    for o in range(n_angles):
        for s in range(n_scales):
            filter = log_gabor_filters[s] * spreads[o]
            eo[o * n_scales + s] = np.fft.ifft2(filter * img_fft.real)
    for i, img1 in enumerate(eo):
        eo[i] = np.abs(eo[i])
        eo[i] = (eo[i] - np.min(eo[i])) / (np.max(eo[i]) - np.min(eo[i]))
        eo[i] = eo[i] * 255
    return eo


def log_gabor(img):
    # print(type(img))
    transf = transforms.ToTensor()
    img = transf(img)
    n_scales = 5
    n_angles = 4
    C, H, W = img.shape
    min_wavelength = 6
    multiplier = 2
    sigma_onf = 0.55
    dThetaOnSigma = 1.2
    thetaSigma = np.pi / n_angles / dThetaOnSigma

    blue, red, green = torch.split(img, split_size_or_sections=1, dim=0)
    blue = torch.squeeze(blue)
    red = torch.squeeze(red)
    green = torch.squeeze(green)
    blue_fft = np.fft.fft2(blue)
    red_fft = np.fft.fft2(red)
    green_fft = np.fft.fft2(green)

    # 定义频率和角度网格
    xrange = np.linspace(-0.5, 0.5, W)
    yrange = np.linspace(-0.5, 0.5, H)
    x, y = np.meshgrid(xrange, yrange)
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)

    # numpy.fft模块中的fftshift函数可以将FFT输出中的直流分量移动到频谱的中央。ifftshift函数则是其逆操作
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    lp_filter = lowpassfilter(H, W, 0.45, 15)

    # 创建Log-Gabor滤波器集合：
    log_gabor_filters = np.zeros((n_scales, H, W))
    #  不同尺度
    for sc in range(n_scales):
        wavelength = min_wavelength * multiplier ** sc
        log_gabor_filters[sc] = np.exp(
            (-(np.log(radius * wavelength + 1e-5)) ** 2) / (2 * np.log(sigma_onf + 1e-5) ** 2)) * lp_filter

    #  创建方向滤波器集合：
    spreads = np.zeros((n_angles, H, W))
    for o in range(n_angles):
        angle = o * np.pi / n_angles
        ds = sintheta * np.cos(angle) - costheta * np.sin(angle)
        dc = costheta * np.cos(angle) + sintheta * np.sin(angle)
        dtheta = np.abs(np.arctan2(ds, dc))
        spreads[o] = np.exp((-dtheta ** 2) / (2 * thetaSigma ** 2))

    #  构造集合的filter
    filter_bank = np.zeros((n_scales * n_angles, H, W))
    eo_b = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_r = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_g = np.zeros((filter_bank.shape[0], filter_bank.shape[1], filter_bank.shape[2]))
    eo_b = filtering(n_angles, n_scales, eo_b, blue_fft, log_gabor_filters, spreads)
    eo_r = filtering(n_angles, n_scales, eo_r, red_fft, log_gabor_filters, spreads)
    eo_g = filtering(n_angles, n_scales, eo_g, green_fft, log_gabor_filters, spreads)

    n_temp_b1 = (eo_b[0] + eo_b[5] + eo_b[10] + eo_b[15]) / 4
    n_temp_r1 = (eo_r[0] + eo_r[5] + eo_r[10] + eo_r[15]) / 4
    n_temp_g1 = (eo_g[0] + eo_g[5] + eo_g[10] + eo_g[15]) / 4
    a1 = np.dstack((n_temp_r1, n_temp_g1, n_temp_b1))
    # a1 = np.transpose(a1, (0, 1, 2))

    n_temp_b2 = (eo_b[2] + eo_b[7] + eo_b[12] + eo_b[17]) / 4
    n_temp_r2 = (eo_r[2] + eo_r[7] + eo_r[12] + eo_r[17]) / 4
    n_temp_g2 = (eo_g[2] + eo_g[7] + eo_g[12] + eo_g[17]) / 4
    a2 = np.dstack((n_temp_r2, n_temp_g2, n_temp_b2))
    # a2 = np.transpose(a2, (0, 1, 2))

    n_temp_b3 = (eo_b[4] + eo_b[9] + eo_b[14] + eo_b[19]) / 4
    n_temp_r3 = (eo_r[4] + eo_r[9] + eo_r[14] + eo_r[19]) / 4
    n_temp_g3 = (eo_g[4] + eo_g[9] + eo_g[14] + eo_g[19]) / 4

    a3 = np.dstack((n_temp_r3, n_temp_g3, n_temp_b3))

    # a3 = np.transpose(a3, (0, 1, 2))
    # max_v1 = a1.max()
    # min_v1 = a1.min()
    # a1 = (a1 - min_v1) / (max_v1 - min_v1)
    #
    # max_v2 = a2.max()
    # min_v2 = a2.min()
    # a2 = (a2 - min_v2) / (max_v2 - min_v2)
    #
    # max_v3 = a3.max()
    # min_v3 = a3.min()
    # a3 = (a3 - min_v3) / (max_v3 - min_v3)

    # a1 = torch.tensor(a1)
    # a2 = torch.tensor(a2)
    # a3 = torch.tensor(a3)
    a1 = a1.astype(np.float32)
    return a1


def log_gabor_3(img):
    img = img.cpu()
    l = img.shape[0]
    out1 = torch.empty(l, 3, 224, 224)
    out2 = torch.empty(l, 3, 224, 224)
    out3 = torch.empty(l, 3, 224, 224)
    for idx, item in enumerate(img):
        a1, a2, a3 = log_gabor(item)
        a1 = torch.transpose(a1, 2, 0)
        a2 = torch.transpose(a2, 2, 0)
        a3 = torch.transpose(a3, 2, 0)
        out1[idx] = a1
        out2[idx] = a2
        out3[idx] = a3
    out1 = out1.to(device)
    out2 = out2.to(device)
    out3 = out3.to(device)
    # out1 = norm(out1)
    # out1 = norm(out1)
    # out1 = norm(out1)

    # out1 = out1.half()
    # out2 = out2.half()
    # out3 = out3.half()
    # out1 = out1.to(torch.float32)
    # out2 = out2.to(torch.float32)
    # out3 = out3.to(torch.float32)
    return out1, out2, out3
