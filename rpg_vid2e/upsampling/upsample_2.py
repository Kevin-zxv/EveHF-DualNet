import argparse
import os
# Must be set before importing torch.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from utils import upsampler_2


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, default='example/original/seq3',help='Path to input directory. See README.md for expected structure of the directory.')
    parser.add_argument("--output_dir", required=True, default='example/upsampled/seq3',help='Path to non-existing output directory. This script will generate the directory.')
    args = parser.parse_args()
    return args


def main():
    flags = get_flags()

    upsampler = upsampler_2.Upsampler(input_dir=flags.input_dir, output_dir=flags.output_dir)
    upsampler.upsample()
    #upsampler.save_upsample_counts()


if __name__ == '__main__':
    main()
