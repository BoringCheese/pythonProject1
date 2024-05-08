import os
import argparse
import random
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
from my_dataset import MyDataSet
from model import mobile_vit_xx_small as create_model
from utils import read_split_data, train_one_epoch, evaluate, log_gabor

seed = random.randint(0, 99999999)

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    print(device)
    g = torch.Generator()
    g1 = torch.Generator()
    g.manual_seed(0)
    g1.manual_seed(0)

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data("H:\\deep-learning-for"
                                                                                               "-image-processing-master"
                                                                                               "\\data_set\\flower_data"
                                                                                               "\\flower_photos")

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     # transforms.Lambda(log_gabor),
                                     transforms.ToTensor(),
                                     # transforms.Lambda(to_float_tensor),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   # transforms.Lambda(log_gabor),
                                   transforms.ToTensor(),
                                   # transforms.Lambda(to_float_tensor),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               generator=g,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn
                                               )
    train_loader1 = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                generator=g1,
                                                pin_memory=True,
                                                num_workers=nw,
                                                collate_fn=train_dataset.collate_fn
                                                )
    # print(type(train_loader))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn
                                             )

    la = torch.empty(0)
    for step, data in enumerate(train_loader):
        images, labels = data
        la = torch.cat((la, labels), dim=0)
    print(la)
    lb = torch.empty(0)
    for step, data in enumerate(train_loader1):
        images, labels = data
        lb = torch.cat((lb, labels), dim=0)
    print(lb)
