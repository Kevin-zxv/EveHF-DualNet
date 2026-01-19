import os
import random

def split_dataset(txt_file1, txt_file2, train_txt, test_txt, split_ratio=0.9):
    """
    将两个 txt 文件中的标签按比例切分为训练集和测试集
    :param txt_file1: 第一个 txt 文件路径
    :param txt_file2: 第二个 txt 文件路径
    :param train_txt: 训练集保存路径
    :param test_txt: 测试集保存路径
    :param split_ratio: 训练集比例，默认为 0.9
    """
    # 读取两个 txt 文件中的数据
    data = []
    for txt_file in [txt_file1, txt_file2]:
        with open(txt_file, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                data.append((img_path, label))

    # 打乱数据
    random.shuffle(data)

    # 按比例切分
    split_index = int(len(data) * split_ratio)
    train_data = data[:split_index]
    test_data = data[split_index:]

    # 保存训练集
    with open(train_txt, 'w') as f:
        for img_path, label in train_data:
            f.write(f"{img_path} {label}\n")

    # 保存测试集
    with open(test_txt, 'w') as f:
        for img_path, label in test_data:
            f.write(f"{img_path} {label}\n")

    print(f"训练集已保存到: {train_txt}，样本数: {len(train_data)}")
    print(f"测试集已保存到: {test_txt}，样本数: {len(test_data)}")


# 示例调用
txt_file1 = r"E:\Desktop\Two_stream_Deep_Video_frame_Detection\clean_DVFI\labels\original_labels.txt"  # 第一个 txt 文件
txt_file2 = r"E:\Desktop\Two_stream_Deep_Video_frame_Detection\clean_DVFI\labels\adacof25_60_labels.txt"  # 第二个 txt 文件
train_txt = 'train.txt'  # 训练集保存路径
test_txt = 'test.txt'    # 测试集保存路径

# 按 9:1 比例切分数据
split_dataset(txt_file1, txt_file2, train_txt, test_txt, split_ratio=0.9)