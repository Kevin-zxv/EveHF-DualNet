import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import argparse
import sys
sys.path.append("/home/Dynamic-convolution-Pytorch-master/make_label")  # 替换为实际路径

# 导入自定义模块
import make_label.dy_11 as dy_11
import make_label.No3_dataset_class as dataset_class
import make_label.combine_net_event as combine_net_event
import make_label.No3_dataset_dualstream as dataset_dualstream

# 配置日志
logging.basicConfig(
    filename='training_continue.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 参数设置
parser = argparse.ArgumentParser(description='Dynamic Convolution Training')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--test-batch-size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--resume-path', type=str, default='best_model_class2_adacof25_60.pth', 
                    help='Path to the checkpoint file to resume from')
args = parser.parse_args()
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据集路径
train_file_pairs = [
    ("/root/autodl-tmp/adacof25_60_6frame.txt", "/root/autodl-tmp/adacof25_60_event6frame.txt"),
    ("/root/autodl-tmp/original_6frame.txt", "/root/autodl-tmp/original_event6frame.txt"),
]

# 创建数据集
traindataset = dataset_dualstream.DualStreamDataset(train_file_pairs, transform=transform)
train_size = int(0.9 * len(traindataset))
test_size = len(traindataset) - train_size
train_dataset, test_dataset = random_split(traindataset, [train_size, test_size])

# 数据加载器
trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True)
testloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True)

# 自定义损失函数
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight)
        pt = torch.exp(-ce_loss)
        return (1 - pt) ** self.gamma * ce_loss

# 初始化模型
model = combine_net_event.DualCombinedModel(
    in_channels=18, 
    in_channels_2=15,
    num_classes=2, 
    convnext_type='base', 
    convnext_type_2='small'
)

# 多GPU支持
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model = model.to(args.device)

# 损失函数和优化器
criterion = FocalLoss(gamma=5)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

# 学习率调整
def adjust_lr(optimizer, epoch):
    if epoch in [args.epochs * 0.5, args.epochs * 0.75, args.epochs * 0.85]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            print(f"LR reduced to {param_group['lr']}")

# 保存检查点
def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

# 加载检查点（修改后的版本）
def load_checkpoint(model, optimizer, device, resume_path):
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        
        # 处理多GPU情况下的状态字典
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        else:
            model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
        
        # 如果检查点包含优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 获取epoch和best_acc（如果存在）
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.)
        
        print(f"Loaded checkpoint from '{resume_path}'")
        print(f"Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")
        return start_epoch, best_acc
    else:
        print(f"No checkpoint found at '{resume_path}', starting from scratch")
        return 0, 0.

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0.
    correct = 0
    
    for batch_idx, (img1, img2, target) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")):
        img1, img2, target = img1.to(args.device), img2.to(args.device), target.to(args.device)
        
        optimizer.zero_grad()
        output = model(img1, img2)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(trainloader.dataset)
    avg_loss = train_loss / len(trainloader)
    print(f'Train Epoch: {epoch} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')
    logging.info(f'Train Epoch: {epoch} | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')

# 验证函数
def validate(epoch):
    model.eval()
    test_loss = 0.
    correct = 0
    
    with torch.no_grad():
        for img1, img2, target in tqdm(testloader, desc="Validating"):
            img1, img2, target = img1.to(args.device), img2.to(args.device), target.to(args.device)
            output = model(img1, img2)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(testloader.dataset)
    avg_loss = test_loss / len(testloader)
    print(f'Validation | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')
    logging.info(f'Validation | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}%')
    return accuracy

# 主训练循环
def main():
    start_epoch = 0
    best_acc = 0.
    
    # 总是尝试加载检查点（无论是否指定--resume）
    start_epoch, best_acc = load_checkpoint(model, optimizer if args.resume else None, args.device, args.resume_path)
    
    for epoch in range(start_epoch, args.epochs):
        adjust_lr(optimizer, epoch)
        train(epoch)
        current_acc = validate(epoch)
        
        # 保存最佳模型
        if current_acc > best_acc:
            best_acc = current_acc
            save_path = 'best_model_class2_adacof25_60_finally.pth'
            # 处理多GPU情况下的保存
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path} with acc: {best_acc:.2f}%")
        
        # 定期保存检查点
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, filename=f'checkpoint_epoch_{epoch}.pth')
    
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()