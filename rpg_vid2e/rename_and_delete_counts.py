import os
import shutil

def extract_rename_and_delete_counts(upsampled_dir, output_dir):
    """
    从 upsampled_dir 中提取 upsample_counts_imgs.txt 文件，
    根据子文件夹名称重命名，保存到 output_dir 中，并删除原文件。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历 upsampled_dir 中的所有子文件夹
    for root, dirs, files in os.walk(upsampled_dir):
        for dir_name in dirs:
            # 检查子文件夹名称是否以 .avi_new 结尾
            if dir_name.endswith(".avi_new"):
                # 构造 imgs 文件夹路径
                imgs_dir = os.path.join(root, dir_name, "imgs")
                # 构造 upsample_counts_imgs.txt 文件路径
                counts_file = os.path.join(imgs_dir, "upsample_counts_imgs.txt")

                # 检查文件是否存在
                if os.path.exists(counts_file):
                    # 提取子文件夹名称（去掉 .avi_new）
                    folder_name = dir_name.replace(".avi_new", "")
                    # 构造新的文件名
                    new_filename = f"{folder_name}_upsample_counts.txt"
                    # 构造新的文件路径
                    new_filepath = os.path.join(output_dir, new_filename)

                    # 复制并重命名文件
                    shutil.copy(counts_file, new_filepath)
                    print(f"Copied and renamed: {counts_file} -> {new_filepath}")

                    # 删除原文件
                    os.remove(counts_file)
                    print(f"Deleted original file: {counts_file}")
                else:
                    print(f"File not found: {counts_file}")
        break  # 只遍历一级子文件夹

# 示例调用
upsampled_dir = "/root/autodl-tmp/upsampled_adacof25_120 /"  # 替换为你的 upsampled 文件夹路径
output_dir = "/root/autodl-tmp/counts"  # 替换为你的输出文件夹路径
extract_rename_and_delete_counts(upsampled_dir, output_dir)
