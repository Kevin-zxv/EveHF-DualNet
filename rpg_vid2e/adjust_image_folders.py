# #判断两张图片是否完全一致
# from PIL import Image
#
#
# # 判断两张图片是否完全相同
# def are_images_identical(image1_path, image2_path):
#     img1 = Image.open(image1_path)
#     img2 = Image.open(image2_path)
#
#     # 比较形状和像素值
#     if img1.size == img2.size and list(img1.getdata()) == list(img2.getdata()):
#         return True
#     return False
#
#
# # 示例
# image1_path = "/root/autodl-tmp/original/v_ApplyEyeMakeup_g01_c01.avi_new/imgs/0004.png"
# image2_path = "/root/autodl-tmp/original/v_ApplyEyeMakeup_g01_c01.avi_new/imgs/0005.png"
# print(are_images_identical(image1_path, image2_path))  # True 或 False

# import numpy as np
#
# # 加载.npz文件
# npz_file_path = 'example/events/seq0/0000000001.npz'  # 替换为你的.npz文件路径
# data = np.load(npz_file_path)
# print(data.files)
# array1 = data['p']
# print(array1)

# # 访问特定的数组（假设我们要访问名为'array1'的数组）
# array_to_save = data['array1']
#
# # 将数组保存为txt文件
# txt_file_path = 'output.txt'  # 替换为你想要保存的txt文件路径
# np.savetxt(txt_file_path, array_to_save, fmt='%s')  # fmt参数用于指定输出格式，这里使用字符串格式

############################################################################################
# # #第一步给每个文件增加fps.txt文件
# import os
# import shutil

# # 获取指定目录下所有文件夹的名称
# def get_all_folders(directory):
#     folders = []
#     for entry in os.scandir(directory):
#         if entry.is_dir():  # 检查是否为文件夹
#             folders.append(entry.path)  # 使用完整路径
#     return folders

# # 定义目标目录
# directory = "/root/autodl-tmp/adacof25_120/"

# # 获取所有文件夹的完整路径
# folders = get_all_folders(directory)

# # 循环处理每个文件夹
# for folder in folders:
#     # 获取文件夹名称（去掉路径）
#     folder_name = os.path.basename(folder)

#     # 创建新文件夹名称（与原有文件夹名称相同）
#     new_folder = os.path.join(os.path.dirname(folder), f"{folder_name}_new")

#     # 创建新文件夹
#     os.makedirs(new_folder, exist_ok=True)

#     # 将原始文件夹移动到新文件夹中
#     shutil.move(folder, os.path.join(new_folder, folder_name))

#     # 在新文件夹中创建 fps.txt 并写入数字 25
#     with open(os.path.join(new_folder, "fps.txt"), "w") as f:
#         f.write("120")

#     print(f"已处理文件夹: {folder} -> {new_folder}")

################################################################
# # #改名为imgs
import os

# 定义目标目录
directory = "/root/autodl-tmp/adacof25_120/"  # 替换为你的目标路径


# 获取所有文件夹的完整路径
def get_all_folders(directory):
    folders = []
    for entry in os.scandir(directory):
        if entry.is_dir():  # 检查是否为文件夹
            folders.append(entry.path)
    return folders


# 获取所有文件夹
folders = get_all_folders(directory)

# 循环处理每个文件夹
for folder in folders:
    # 获取文件夹中的所有子文件夹
    for subdir in os.scandir(folder):
        if subdir.is_dir():  # 检查是否为子文件夹
            # 构造新的子文件夹路径（改为 img）
            new_subdir_path = os.path.join(folder, "imgs")

            # 检查是否已存在同名文件夹
            if os.path.exists(new_subdir_path):
                print(f"文件夹 {new_subdir_path} 已存在，跳过重命名")
                continue

            # 重命名子文件夹
            os.rename(subdir.path, new_subdir_path)
            print(f"已将 {subdir.path} 重命名为 {new_subdir_path}")


