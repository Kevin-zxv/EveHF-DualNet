import os
from PIL import Image

def get_first_image_sizes(root_dir):
    """
    检查大文件夹中每个子文件夹的第一张图片的大小。
    :param root_dir: 大文件夹路径
    :return: 一个字典，键为子文件夹路径，值为第一张图片的尺寸
    """
    size_dict = {}  # 存储子文件夹路径及其第一张图片的尺寸
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # 只处理图片文件（支持常见格式）
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                file_path = os.path.join(subdir, file)
                try:
                    with Image.open(file_path) as img:
                        size = img.size  # 获取图片尺寸 (width, height)
                        size_dict[subdir] = size  # 将子文件夹路径和图片尺寸存入字典
                        break  # 处理完第一张图片后，跳出循环
                except Exception as e:
                    print(f"无法处理文件 {file_path}: {e}")
                break  # 处理完第一张图片后，跳出循环
    return size_dict

def find_unique_sizes(size_dict):
    """
    从所有子文件夹的第一张图片尺寸中找出不一样的尺寸。
    :param size_dict: 子文件夹路径及其第一张图片尺寸的字典
    :return: 一个集合，包含所有唯一的尺寸
    """
    # 使用集合存储尺寸（自动去重）
    unique_sizes = set(size_dict.values())
    return unique_sizes

# 示例用法
root_dir = r"/root/autodl-tmp/events/event_original"  # 大文件夹路径
size_dict = get_first_image_sizes(root_dir)

# 找出不一样的尺寸
unique_sizes = find_unique_sizes(size_dict)

# 打印结果
print("所有子文件夹的第一张图片尺寸：")
for subdir, size in size_dict.items():
    print(f"子文件夹: {subdir}, 图片尺寸: {size}")

print("\n不一样的尺寸：")
for size in unique_sizes:
    print(size)

#######################
# from PIL import Image
# import os
#
# # 定义目标尺寸
# target_size = (492, 369)
#
# # 定义图片目录
# image_dir = '/root/autodl-tmp/events/adacof25_60_frame/v_PommelHorse_g05_c01.avi_new'
#
# # 遍历目录中的所有文件
# for filename in os.listdir(image_dir):
#     # 检查文件是否为图片（可以根据需要扩展支持的格式）
#     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
#         # 构建完整的文件路径
#         file_path = os.path.join(image_dir, filename)
#
#         # 打开图片
#         with Image.open(file_path) as img:
#             # 调整图片尺寸
#             resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
#
#             # 保存调整后的图片（覆盖原文件或保存为新文件）
#             resized_img.save(file_path)
#
# print("所有图片已调整尺寸！")