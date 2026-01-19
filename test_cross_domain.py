import torch.optim as optim
import sys
sys.path.append("/home/Dynamic-convolution-Pytorch-master/make_label")  # 替换为实际路径
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import argparse
import make_label.dy_11 as dy_11
import dataset
import torchvision.transforms as transforms
from tqdm import tqdm  # 导入 tqdm
import logging

import make_label.No3_dataset_class as dataset_class
# import make_label.combine_net as combine_net
import make_label.combine_net_event as combine_net_event
from torch.utils.data import Dataset, DataLoader, random_split
import make_label.No3_dataset_dualstream as dataset_dualstream

# 加载模型
model = combine_net_event.DualCombinedModel(in_channels=18, in_channels_2=15, num_classes=10, convnext_type='base', convnext_type_2='small')
model.load_state_dict(torch.load('best_model.pth'))
model = model.to('cuda')
model.eval()  # 将模型设置为评估模式
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# 准备新的测试数据集
new_test_file_pairs = [
    ("/root/autodl-tmp/labels/rrin25_60_6frame_labels.txt", "/root/autodl-tmp/labels/eventrrin25_60_6frame_labels.txt"),
    # 添加更多文件对
]

new_test_dataset = dataset_dualstream.DualStreamDataset(new_test_file_pairs, transform=transform_test)
new_test_loader = DataLoader(new_test_dataset, batch_size=64, shuffle=False, num_workers=10, pin_memory=True)

# 测试函数
def test():
    model.eval()
    test_loss = 0.
    correct = 0.
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for image1, image2, label in tqdm(new_test_loader, desc="Testing"):
            image1, image2, label = image1.to('cuda'), image2.to('cuda'), label.to('cuda')
            output = model(image1, image2)
            test_loss += F.cross_entropy(output, label, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            
            probs = F.softmax(output, dim=1)
            all_targets.append(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    test_loss /= len(new_test_loader.dataset)
    accuracy = 100. * correct / len(new_test_loader.dataset)
    
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    try:
        auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        auc = 0.0
        
    print(f'Test Set: Avg Loss: {test_loss:.4f}, AUC: {auc:.4f}')
    logging.info(f'Test Set: Avg Loss: {test_loss:.4f}, AUC: {auc:.4f}')

# 运行测试
test()