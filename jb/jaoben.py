import os
from array import array
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from jb.utils import log_gabor1
from jb.lg import l_g1


def process_image(image_path):
    """将图片转换为灰度图，并保存到输出文件夹"""
    # 转换图片
    g_image = l_g1(image_path)
    g_image = (g_image - g_image.min()) / (g_image.max() - g_image.min()) * 255
    g_image = g_image.astype(int)

    g_image = g_image.astype(np.uint8)
    g_image = Image.fromarray(g_image)
    return g_image


def process_folder(input_folder, output_folder):
    """
    处理指定文件夹内的所有子文件夹中的图片，并将处理后的图片保存到输出文件夹的对应子文件夹中。
    """
    # 遍历输入文件夹内的子文件夹
    for subdir in os.listdir(input_folder):
        subdir_path = os.path.join(input_folder, subdir)
        if os.path.isdir(subdir_path):
            output_subdir = os.path.join(output_folder, subdir)
            # 确保输出的子文件夹存在
            if os.path.exists(output_subdir):
                continue
            else:
                os.makedirs(output_subdir)

            # 遍历子文件夹内的图片
            for filename in os.listdir(subdir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(subdir_path, filename)

                    # 处理图片
                    processed_image = process_image(image_path)

                    # 保存处理后的图片到输出文件夹的对应子文件夹
                    output_path = os.path.join(output_subdir, filename)
                    processed_image.save(output_path)


if __name__ == '__main__':
    # 指定输入和输出文件夹路径
    input_folder = 'H:\\Img\\train'
    output_folder = 'H:\\Img_g\\train'
    # 处理文件夹
    process_folder(input_folder, output_folder)
