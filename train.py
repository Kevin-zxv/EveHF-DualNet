import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import argparse
import dynamic_models.dy_11
import dataset
import torchvision.transforms as transforms
from tqdm import tqdm  # 导入 tqdm
import logging
import make_label.No3_dataset_class as dataset_class
import make_label.combine_net as combine_net
from torch.utils.data import Dataset, DataLoader
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

transform_train = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),          # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])


transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])


# train_txt_file_list = [
#         r"/root/autodl-tmp/event_train6.txt",
#     ]
# test_txt_file_list = [
#         r"/root/autodl-tmp/event_test6.txt",
#     ]
train_txt_file_list = [
        r"/root/autodl-tmp/video_frame/train6.txt",
    ]
test_txt_file_list = [
        r"/root/autodl-tmp/video_frame/test6.txt",
    ]

trainset = dataset_class.MultiFileMultiFrameDataset(
    file_list=train_txt_file_list,
    transform=transform_train,
)
testset = dataset_class.MultiFileMultiFrameDataset(
    file_list=test_txt_file_list,
    transform=transform_test,
)

trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=10,pin_memory=True)
testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=10,pin_memory=True)

model=combine_net.CombinedModel(in_channels=18, num_classes=10,convnext_type='base')
# 检查是否有多个 GPU
if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 张 GPU")
    model = nn.DataParallel(model)
# 将模型移动到 GPU
model = model.to('cuda')

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
    for batch_idx, (data, target) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
        data, target = data.to(args.device), target.to(args.device)
        # print(f"Batch {batch_idx + 1}")
        # print(f"Images shape: {data.shape}")  # 打印图片的维度
        # print(f"Labels shape: {target.shape}")  # 打印标签的维度
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
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
        for data, label in tqdm(testloader, desc="Validating"):
            data, label = data.to(args.device), label.to(args.device)
            output = model(data)
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