# EveHF-DualNet Training Guide

This guide provides instructions on how to configure and run the training script `train.py` for the EveHF-DualNet project.

## Prerequisites

Ensure you have the following installed:
- Python 3.x
- PyTorch (with CUDA support recommended)
- torchvision
- tqdm

You also need the following custom modules (ensure they are in your python path or project directory):
- `make_label` (containing `No3_dataset_class` and `combine_net`)
- `dynamic_models`

## Data Preparation

For time frame generation and video-to-event conversion, please refer to: [https://github.com/uzh-rpg/rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e)

The training script expects data file paths to be provided in text files. You need to prepare two text files:
1.  **Training List**: A text file containing paths to training images/frames.
2.  **Testing List**: A text file containing paths to testing images/frames.

The format of these text files should match what `make_label.No3_dataset_class.MultiFileMultiFrameDataset` expects (typically `path/to/image label`).

## Configuration

Before running the script, you must update the file paths in `train.py` to point to your actual data list files.

Open `train.py` and modify lines 49-54:

```python
train_txt_file_list = [
    r"/path/to/your/train_list.txt",  # Update this path
]
test_txt_file_list = [
    r"/path/to/your/test_list.txt",   # Update this path
]
```

## Usage

Run the training script using Python. You can customize the training parameters using command-line arguments.

### Basic Command

```bash
python train.py
```

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | `fliter` | Name of the training dataset. |
| `--batch-size` | int | `64` | Batch size for training. |
| `--test-batch-size` | int | `20` | Batch size for testing/validation. |
| `--epochs` | int | `160` | Number of training epochs. |
| `--lr` | float | `0.01` | Learning rate. |
| `--momentum` | float | `0.9` | SGD momentum. |
| `--weight-decay` | float | `1e-4` | Weight decay. |
| `--net-name` | str | `dy_resnet18`| Network name (informational). |

### Example with Custom Arguments

```bash
python train.py --batch-size 32 --epochs 100 --lr 0.005
```

## Output

- **Logs**: Training progress is saved to `training.log`.
- **Model Checkpoint**: The best model (based on validation accuracy) is saved as `best_model.pth`.
