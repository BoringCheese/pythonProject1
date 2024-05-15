import os
from PIL import Image

# 设置包含1000个子文件夹的主文件夹路径
main_folder_path = 'D:\\img\\ImageNet\\train'

# 遍历主文件夹中的每个子文件夹
for folder_name in os.listdir(main_folder_path):
    folder_path = os.path.join(main_folder_path, folder_name)

    # 确保当前路径是文件夹
    if os.path.isdir(folder_path):
        # 遍历文件夹内的所有文件
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            t = 0
            # 尝试打开图像文件
            try:
                with Image.open(file_path) as img:
                    # 检查图像是否为RGB模式
                    if img.mode != 'RGB':
                        t = 1
                        # 如果不是RGB模式，则删除图像文件
            except IOError:
                # 如果文件不是图像或无法打开，忽略错误
                print(f"Failed to open or process file: {file_path}")
            if t == 1:
                os.remove(file_path)
                print(f"Deleted non-RGB image: {file_path}")
print("Cleanup complete.")
