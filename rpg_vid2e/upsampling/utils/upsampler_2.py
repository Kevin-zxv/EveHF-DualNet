import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from . import Sequence
from .const import imgs_dirname
from .interpolator import Interpolator
from .utils import get_sequence_or_none


class Upsampler:
    _timestamps_filename = 'timestamps.txt'

    def __init__(self, input_dir: str, output_dir: str):
        assert os.path.isdir(input_dir), 'The input directory must exist'
        assert not os.path.exists(output_dir), 'The output directory must not exist'

        self._prepare_output_dir(input_dir, output_dir)
        self.src_dir = input_dir
        self.dest_dir = output_dir

        path = os.path.join(os.path.dirname(__file__), "../../pretrained_models/film_net/Style/saved_model")
        self.interpolator = Interpolator(path, None)
        self.upsample_counts_dict = {}  # 初始化字典

    def upsample(self):
        sequence_counter = 0
        for src_absdirpath, dirnames, filenames in os.walk(self.src_dir):
            sequence = get_sequence_or_none(src_absdirpath)
            if sequence is None:
                continue
            sequence_counter += 1
            print('Processing sequence number {}'.format(src_absdirpath))
            reldirpath = os.path.relpath(src_absdirpath, self.src_dir)
            dest_imgs_dir = os.path.join(self.dest_dir, reldirpath, imgs_dirname)
            dest_timestamps_filepath = os.path.join(self.dest_dir, reldirpath, self._timestamps_filename)
            self.upsample_sequence(sequence, dest_imgs_dir, dest_timestamps_filepath)
        # 保存所有文件夹的计数
        self.save_upsample_counts()

    def save_upsample_counts_2(self):
        counts_filepath = os.path.join(self.dest_dir, 'upsample_counts.txt')
        with open(counts_filepath, 'w') as f:
            for count in self.upsample_counts:
                f.write(f"{count}\n")

    def save_upsample_counts(self):
        """
        保存每个文件夹的上采样计数到独立的文件中。
        """
        for dest_imgs_dir, counts in self.upsample_counts_dict.items():
            # 获取当前文件夹的名称
            parent_dir_name = os.path.basename(os.path.normpath(dest_imgs_dir))
            # 构造计数文件路径
            counts_filepath = os.path.join(dest_imgs_dir, f'upsample_counts_{parent_dir_name}.txt')
            # 写入计数
            with open(counts_filepath, 'w') as f:
                for count in counts:
                    f.write(f"{count}\n")

    def upsample_sequence(self, sequence: Sequence, dest_imgs_dir: str, dest_timestamps_filepath: str):
        os.makedirs(dest_imgs_dir, exist_ok=True)
        timestamps_list = list()

        # 初始化当前文件夹的计数列表
        self.upsample_counts_dict[dest_imgs_dir] = []

        idx = 0
        for img_pair, time_pair in tqdm(next(sequence), total=len(sequence), desc=type(sequence).__name__):
            I0 = img_pair[0][None]
            I1 = img_pair[1][None]
            t0, t1 = time_pair
            total_frames, total_timestamps, count = self._upsample_adaptive(I0, I1, t0, t1)
            total_frames = [I0[0]] + total_frames
            timestamps = [t0] + total_timestamps
            self.upsample_counts_dict[dest_imgs_dir].append(count)  # 更新当前文件夹的计数
            sorted_indices = np.argsort(timestamps)
            total_frames = [total_frames[j] for j in sorted_indices]
            timestamps = [timestamps[i] for i in sorted_indices]
            timestamps_list += timestamps
            for frame in total_frames:
                self._write_img(frame, idx, dest_imgs_dir)
                idx += 1

        timestamps_list.append(t1)
        self._write_img(I1[0, ...], idx, dest_imgs_dir)
        self._write_timestamps(timestamps_list, dest_timestamps_filepath)

    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections=-1):
        if num_bisections == 0:
            image, _, _ = self.interpolator.interpolate(I0, I1, np.array([0.5], dtype=np.float32))
            return [image[0]], [(t0 + t1) / 2], 1

        dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        image, F_0_1, F_1_0 = self.interpolator.interpolate(I0, I1, dt)

        if num_bisections < 0:
            flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** 0.5).max()
            flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** 0.5).max()
            num_bisections = int(np.ceil(np.log(max(flow_mag_0_1_max, flow_mag_1_0_max)) / np.log(2)))

            if num_bisections == 0:
                return [image[0]], [(t0 + t1) / 2], 1

        left_images, left_timestamps, left_count = self._upsample_adaptive(I0, image, t0, (t0 + t1) / 2, num_bisections - 1)
        right_images, right_timestamps, right_count = self._upsample_adaptive(image, I1, (t0 + t1) / 2, t1, num_bisections - 1)
        images = left_images + [image[0]] + right_images
        timestamps = left_timestamps + [(t0 + t1) / 2] + right_timestamps
        count = left_count + 1 + right_count
        return images, timestamps, count


    def _prepare_output_dir(self, src_dir: str, dest_dir: str):
            # Copy directory structure.
            def ignore_files(directory, files):
                return [f for f in files if os.path.isfile(os.path.join(directory, f))]
            shutil.copytree(src_dir, dest_dir, ignore=ignore_files)

    @staticmethod
    def _write_img(img: np.ndarray, idx: int, imgs_dir: str):
        assert os.path.isdir(imgs_dir)
        img = np.clip(img * 255, 0, 255).astype("uint8")
        path = os.path.join(imgs_dir, "%08d.png" % idx)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, img)

    @staticmethod
    def _write_timestamps(timestamps: list, timestamps_filename: str):
        with open(timestamps_filename, 'w') as t_file:
            t_file.writelines([str(t) + '\n' for t in timestamps])