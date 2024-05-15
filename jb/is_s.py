import os

# 设置文件夹路径
folder1 = 'D:/ImageNet/train'
folder2 = 'D:/ImageNet_G/train'

# 读取两个文件夹中的子文件夹名字
subfolders1 = os.listdir(folder1)
subfolders2 = os.listdir(folder2)

# 确保两个文件夹中的子文件夹完全相同
if set(subfolders1) != set(subfolders2):
    print(f"子文件夹名称不匹配{subfolders1}{subfolders2}")
else:
    # 遍历每一个子文件夹，比较文件数量
    for subfolder in subfolders1:
        # 获取每个子文件夹的文件列表
        files1 = os.listdir(os.path.join(folder1, subfolder))
        files2 = os.listdir(os.path.join(folder2, subfolder))

        # 比较文件数量
        if len(files1) != len(files2):
            print(f"文件夹 {subfolder} 的文件数量不一致: {len(files1)} vs {len(files2)}")
        # else:
        #     print(f"文件夹 {subfolder} 的文件数量一致: {len(files1)}")
