import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        """
        初始化数据集
        :param txt_file: 包含图像路径和标签的 txt 文件
        :param transform: 数据预处理
        """
        self.transform = transform
        self.img_paths = []
        self.labels = []

        # 从 txt 文件中读取图像路径和标签
        with open(txt_file, 'r') as f:
            for line in f:
                img_path, label = line.strip().split()
                self.img_paths.append(img_path)  # 直接使用完整路径
                self.labels.append(int(label))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]  # 直接使用完整路径
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# 示例调用
if __name__ == "__main__":
    # 定义 txt 文件路径
    txt_file = 'train.txt'  # 包含图像路径和标签的 txt 文件

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 创建数据集实例
    dataset = CustomDataset(txt_file=txt_file, transform=transform)

    # 使用 DataLoader 加载数据
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 检查数据加载是否正常
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print("Images shape:", images.shape)
        print("Labels:", labels)
        break  # 只检查第一个批次