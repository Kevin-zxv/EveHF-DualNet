# 2025.2.14最新
import argparse
import glob
import numpy as np
import os
import matplotlib.pyplot as plt


def render(x, y, t, p, shape):
    img = np.full(shape=shape + [3], fill_value=255, dtype="uint8")
    img[y, x, :] = 0
    #img[y, x, p+1] = 255
    p = [int(x) + 1 for x in p]
    img[y, x, p] = 255
    return img

'''
def parse_segments(index_file):
    """
    从 .txt 文件中解析分段结束点，并生成分段范围
    """
    segments = []
    with open(index_file, "r") as f:
        end_points = [int(line.strip()) for line in f if line.strip()]

    # 生成分段范围
    start = 0
    for end in end_points:
        segments.append((start, end))
        start = end + 1
    return segments
'''

def parse_segments(index_file):
    """
    从 .txt 文件中解析分段长度，并生成分段范围

    参数:
        index_file (str): 包含分段长度的 .txt 文件路径

    返回:
        list: 分段范围列表，每个元素为 (start, end)
    """
    segments = []  # 用于存储分段范围
    with open(index_file, "r") as f:
        # 读取文件，逐行解析为整数，并过滤空行
        segment_lengths = [int(line.strip()) for line in f if line.strip()]

    # 生成分段范围
    start = 0  # 第一个分段的起始点
    for length in segment_lengths:
        end = start + length - 1  # 当前分段的结束点
        segments.append((start, end))  # 添加分段范围
        start = end + 1  # 更新下一个分段的起始点
    return segments


def process_event_folder(event_folder, segments, output_folder, shape):
    """
    处理一个 event 文件夹中的 .npz 文件
    """
    # 加载事件文件
    event_files = sorted(glob.glob(os.path.join(event_folder, "*.npz")))

    # 遍历每个分段
    for seg_idx, (start, end) in enumerate(segments):
        X = []
        Y = []
        P = []
        T = []

        # 遍历分段范围内的文件
        for index in range(start, end + 1):
            if index >= len(event_files):
                print(f"Warning: Index {index} is out of range. Skipping.")
                continue

            f = event_files[index]
            events = np.load(f)
            x = events['x']
            y = events['y']
            t = events['t']
            p = events['p']

            # 追加数据到列表
            X.extend(x)
            Y.extend(y)
            T.extend(t)
            P.extend(p)

        # 渲染图像
        img = render(x=X, y=Y, p=P, t=T, shape=shape)

        # 保存图像
        output_filename = f"output_image_segment_{seg_idx + 1}.png"
        output_path = os.path.join(output_folder, output_filename)
        plt.imshow(img)
        plt.axis('off')  # 可选：隐藏坐标轴
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # 关闭图像以释放内存
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("""Generate events from a high frequency video stream""")
    parser.add_argument("--txt_dir", default="/root/autodl-tmp/counts", help="Path to the directory containing .txt files")
    parser.add_argument("--event_dir", default="/root/autodl-tmp/events", help="Path to the directory containing event folders")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/output", help="Path to the output directory")
    parser.add_argument("--shape", nargs=2, default=[240, 320], type=int, help="Shape of the output image (height, width)")
    #parser.add_argument("--shape", nargs=2, default=[226, 400], type=int, help="Shape of the output image (height, width)")
    args = parser.parse_args()

    # 创建输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)

    # 遍历 .txt 文件
    txt_files = sorted(glob.glob(os.path.join(args.txt_dir, "*.txt")))
    for txt_file in txt_files:
        # 解析分段范围
        segments = parse_segments(txt_file)

        # 获取对应的 event 文件夹名
        txt_name = os.path.basename(txt_file)  # 示例：v_ApplyEyeMakeup_g01_c01_upsample_counts.txt
        base_name = os.path.splitext(txt_name)[0]  # 示例：v_ApplyEyeMakeup_g01_c01_upsample_counts

        # 关键修改点：从文件名中提取共同部分
        event_folder_part = base_name.replace("_upsample_counts", "")
        event_folder_name = f"{event_folder_part}.avi_new"
        event_folder = os.path.join(args.event_dir, event_folder_name)

        if not os.path.exists(event_folder):
            print(f"Warning: Event folder {event_folder} does not exist. Skipping.")
            continue

        # 创建输出子文件夹
        output_folder = os.path.join(args.output_dir, event_folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # 处理 event 文件夹
        print(f"Processing: {txt_file} -> {event_folder}")
        process_event_folder(event_folder, segments, output_folder, args.shape)

    print("All files processed successfully!")
