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
parser = argparse.ArgumentParser(description='dynamic convolution')
parser.add_argument('--dataset', type=str, default='fliter', help='training dataset')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--test-batch-size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--net-name', default='dy_resnet18')

args = parser.parse_args()
args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),          # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])


transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

train_file_pairs = [
    ("/root/autodl-tmp/labels/original_6frame.txt", "/root/autodl-tmp/labels/originalevent_6frame.txt"),
    ("/root/autodl-tmp/labels/adacof25_30_6frame.txt","/root/autodl-tmp/labels/adacof25_30event_6frame.txt"),
    ("/root/autodl-tmp/labels/adacof25_60_6frame.txt","/root/autodl-tmp/labels/adacof25_60event_6frame.txt"),
    # 添加更多文件对
]

test_file_pairs = [
    ("/root/autodl-tmp/labels/rrin25_60_6frame_labels.txt", "/root/autodl-tmp/labels/eventrrin25_60_6frame_labels.txt"),
    # 添加更多文件对
]

traindataset = dataset_dualstream.DualStreamDataset(train_file_pairs, transform=transform)
# testdataset=dataset_dualstream.DualStreamDataset(test_file_pairs, transform=transform_test)
# 划分训练集和测试集（例如 80% 训练集，20% 测试集）
train_size = int(0.9 * len(traindataset))  # 训练集大小
test_size = len(traindataset) - train_size   # 测试集大小
train_dataset, test_dataset = random_split(traindataset, [train_size, test_size])

# 创建 DataLoader
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=10,pin_memory=True)  # 训练集需要 shuffle
testloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,num_workers=10,pin_memory=True)  # 测试集不需要 shuffle
# trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10,pin_memory=True)
# testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10,pin_memory=True)

# model=combine_net.CombinedModel(in_channels=18, num_classes=10,convnext_type='base')
model=combine_net_event.DualCombinedModel(in_channels=18, in_channels_2=15,num_classes=10, convnext_type='base',convnext_type_2='small')
# 检查是否有多个 GPU
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU")
    model = nn.DataParallel(model)
# 将模型移动到 GPU
model = model.to('cuda')
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_ap = torch.norm(anchor - positive, p=2, dim=1)  # 锚点与正样本的距离
        d_an = torch.norm(anchor - negative, p=2, dim=1)  # 锚点与负样本的距离
        loss = torch.relu(d_ap - d_an + self.margin)  # 三元组损失
        return loss.mean()

# 初始化三元组损失函数
criterion_triplet = TripletLoss(margin=3.0)
# 优化器
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# 学习率调整
def adjust_lr(optimizer, epoch):
    if epoch in [args.epochs * 0.5, args.epochs * 0.75, args.epochs * 0.85]:
        for p in optimizer.param_groups:
            p['lr'] *= 0.1
            lr = p['lr']
        print('Change lr:' + str(lr))

# 训练函数
# 训练函数
# 配置日志
logging.basicConfig(
    filename='training.log',  # 日志文件名
    level=logging.INFO,       # 日志级别
    format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式
)
def train(epoch):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    adjust_lr(optimizer, epoch)

    # 使用 tqdm 包装 trainloader
    for batch_idx, (image1, image2, target) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
        image1, image2, target = image1.to(args.device), image2.to(args.device),target.to(args.device)
        optimizer.zero_grad()
        # 前向传播
        features1,features2,output = model(image1, image2)
        anchor_idx1 = torch.randint(0, len(features1), (1,),device=args.device).item()  # 随机选择一个索引
        anchor_idx2 = torch.randint(0, len(features2), (1,)).item()  # 随机选择一个索引
        anchor1 = features1[anchor_idx1].unsqueeze(0)  # 提取锚点并增加批次维度
        anchor2 = features2[anchor_idx2].unsqueeze(0)
        # positive1 = features1[target == target[0]]  # 选择同类样本作为正样本
        # negative1 = features1[target != target[0]]  # 选择不同类样本作为负样本
        # positive2 = features2[target == target[0]]  # 选择同类样本作为正样本
        # negative2 = features2[target != target[0]]  # 选择不同类样本作为负样本
        positive_mask1 = (target == target[anchor_idx1]) & (torch.arange(len(features1),device=args.device) != anchor_idx1)
        positive1 = features1[positive_mask1]
        positive_mask2 = (target == target[anchor_idx2]) & (torch.arange(len(features2),device=args.device) != anchor_idx2)
        positive2 = features2[positive_mask2]

        negative_mask1 = (target != target[anchor_idx1])
        negative1 = features1[negative_mask1]
        negative_mask2 = (target != target[anchor_idx2])
        negative2 = features2[negative_mask2]
        if len(positive1) == 0 or len(negative1) == 0 or len(positive2) == 0 or len(negative2) == 0:
            continue
        # 随机选择正样本和负样本
        positive1 = positive1[torch.randint(0, len(positive1), (1,),device=args.device)].unsqueeze(0)
        negative1 = negative1[torch.randint(0, len(negative1), (1,),device=args.device)].unsqueeze(0)
        positive2 = positive2[torch.randint(0, len(positive2), (1,),device=args.device)].unsqueeze(0)
        negative2 = negative2[torch.randint(0, len(negative2), (1,),device=args.device)].unsqueeze(0)

        # 计算三元组损失
        loss_triplet1 = criterion_triplet(anchor1, positive1, negative1)
        loss_triplet2 = criterion_triplet(anchor2, positive2, negative2)
        loss_ce = F.cross_entropy(output, target)
        loss=loss_triplet1 + loss_triplet2 + loss_ce
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        loss.backward()
        optimizer.step()

    print(f'Train Epoch: {epoch}, Loss: {loss.item():.6f}, Acc: {train_acc / len(trainloader.dataset):.4f}')
    logging.info(f'Train Epoch: {epoch}, Loss: {loss.item():.6f}, Acc: {train_acc / len(trainloader.dataset):.4f}')


# 验证函数
def val(epoch):
    model.eval()
    test_loss = 0.
    correct = 0.

    # 使用 tqdm 包装 testloader
    with torch.no_grad():
        for image1, image2, label in tqdm(testloader, desc="Validating"):
            image1, image2, label = image1.to(args.device), image2.to(args.device),label.to(args.device)
            features1,features2,output = model(image1, image2)
            test_loss += F.cross_entropy(output, label, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    accuracy = 100. * correct / len(testloader.dataset)
    print(f'Test Set: Avg Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    logging.info(f'Test Set: Avg Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

# 训练和验证
best_val_acc = 0.
for epoch in range(args.epochs):
    train(epoch)
    temp_acc = val(epoch)
    if temp_acc > best_val_acc:
        best_val_acc = temp_acc
        torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
print(f'Best Test Accuracy: {best_val_acc:.2f}%')