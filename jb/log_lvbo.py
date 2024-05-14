import os
from PIL import Image
import numpy as np
from utils import log_gabor1


def process_images(source_folder, dest_folder):
    # 遍历源目录中的所有子目录
    for subdir in os.listdir(source_folder):
        src_path = os.path.join(source_folder, subdir)
        if os.path.isdir(src_path):
            dest_path = os.path.join(dest_folder, subdir)
            # 创建目标目录
            os.makedirs(dest_path, exist_ok=True)

            # 处理每个文件
            for filename in os.listdir(src_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(src_path, filename)
                    # 打开并处理图片
                    with Image.open(file_path) as img:
                        processed_image = log_gabor1(img)
                        processed_image = processed_image.astype(np.uint8)
                        img1 = Image.fromarray(processed_image)
                        # 保存处理后的图片到新的目录
                        img1.save(os.path.join(dest_path, filename))
                        print(f'Processed and saved: {os.path.join(dest_path, filename)}')


# 指定源目录和目标目录
source_folder = 'H:\\data_set\\flower_data\\flower_photos'
dest_folder = 'H:\\data_f\\flower_data_l'
process_images(source_folder, dest_folder)
