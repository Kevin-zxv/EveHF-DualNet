import numpy as np


def load_labels(txt_path):
    """从txt文件中提取每行的最后一列标签"""
    with open(txt_path) as f:
        lines = [line.strip().split() for line in f]
    return [int(items[-1]) for items in lines]  # 提取最后一列并转换为整数


def compare_labels(labels1, labels2):
    """比较两组标签是否完全相同"""
    if len(labels1) != len(labels2):
        print("标签数量不一致：{} vs {}".format(len(labels1), len(labels2)))
        return False

    mismatches = []
    for idx, (l1, l2) in enumerate(zip(labels1, labels2)):
        if l1 != l2:
            mismatches.append(idx)

    if mismatches:
        print("发现 {} 处不一致，首处行号：{}".format(len(mismatches), mismatches[0]))
        return False
    else:
        print("所有标签一致")
        return True


def compare_labels_with_log(labels1, labels2, log_path="mismatch.log"):
    """比较标签并保存不一致的行号"""
    mismatches = []
    for idx, (l1, l2) in enumerate(zip(labels1, labels2)):
        if l1 != l2:
            mismatches.append(idx)

    if mismatches:
        with open(log_path, "w") as f:
            f.write("不一致行号：\n")
            f.write("\n".join(map(str, mismatches)))
        print("发现 {} 处不一致，已保存到 {}".format(len(mismatches), log_path))
        return False
    else:
        print("所有标签一致")
        return True


def compare_labels_fast(labels1, labels2):
    """使用numpy加速标签比较"""
    arr1 = np.array(labels1)
    arr2 = np.array(labels2)

    if len(arr1) != len(arr2):
        print("标签数量不一致：{} vs {}".format(len(arr1), len(arr2)))
        return False

    mismatches = np.where(arr1 != arr2)[0]
    if mismatches.size > 0:
        print("发现 {} 处不一致，首处行号：{}".format(mismatches.size, mismatches[0]))
        return False
    else:
        print("所有标签一致")
        return True


def validate_label_consistency(doc1_path, doc2_path, log_path=None, use_numpy=False):
    """验证两个文档的标签一致性"""
    # 加载标签
    labels1 = load_labels(doc1_path)
    labels2 = load_labels(doc2_path)

    # 比较标签
    if use_numpy:
        return compare_labels_fast(labels1, labels2)
    elif log_path:
        return compare_labels_with_log(labels1, labels2, log_path)
    else:
        return compare_labels(labels1, labels2)


# 示例使用
if __name__ == "__main__":
    # 假设文档路径
    doc1_path = "/root/autodl-tmp/labels/adacof25_60_6frame_labels.txt"
    doc2_path = "/root/autodl-tmp/labels/eventdacof25_60_6frame_labels.txt"

    # 验证标签一致性
    if validate_label_consistency(doc1_path, doc2_path, log_path="mismatch.log", use_numpy=True):
        print("两个文档的标签完全一致")
    else:
        print("发现标签不一致，请检查数据")