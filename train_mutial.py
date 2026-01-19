import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from tqdm import tqdm
import logging
import argparse

# 添加自定义模块路径
sys.path.append("/home/Dynamic-convolution-Pytorch-master/make_label")
import make_label.dy_11 as dy_11
import make_label.No3_dataset_class as dataset_class
import make_label.combine_net_event as combine_net_event
import make_label.No3_dataset_dualstream as dataset_dualstream

# 初始化分布式环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

# 清理分布式环境
def cleanup():
    dist.destroy_process_group()

# 配置日志
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 学习率调整
def adjust_lr(optimizer, epoch, args):
    if epoch in [int(args.epochs * 0.5), int(args.epochs * 0.75), int(args.epochs * 0.85)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        print(f'Change lr: {param_group["lr"]}')

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

# 训练函数
def train(epoch, model, train_loader, optimizer, rank, args):
    model.train()
    avg_loss = 0.0
    train_acc = 0.0

    # 使用 DistributedSampler 确保数据分片
    train_loader.sampler.set_epoch(epoch)

    for batch_idx, (image1, image2, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", disable=(rank != 0))):
        image1 = image1.to(rank)
        image2 = image2.to(rank)
        target = target.to(rank)

        optimizer.zero_grad()
        features1, features2, output = model(image1, image2)
        anchor_idx1 = torch.randint(0, len(features1), (1,), device=rank).item()  # 使用 rank 作为设备
        anchor_idx2 = torch.randint(0, len(features2), (1,), device=rank).item()  # 使用 rank 作为设备
        anchor1 = features1[anchor_idx1].unsqueeze(0)  # 提取锚点并增加批次维度
        anchor2 = features2[anchor_idx2].unsqueeze(0)

        # 选择正样本和负样本
        positive_mask1 = (target == target[anchor_idx1]) & (torch.arange(len(features1), device=rank) != anchor_idx1)
        positive1 = features1[positive_mask1]
        positive_mask2 = (target == target[anchor_idx2]) & (torch.arange(len(features2), device=rank) != anchor_idx2)
        positive2 = features2[positive_mask2]

        negative_mask1 = (target != target[anchor_idx1])
        negative1 = features1[negative_mask1]
        negative_mask2 = (target != target[anchor_idx2])
        negative2 = features2[negative_mask2]

        if len(positive1) == 0 or len(negative1) == 0 or len(positive2) == 0 or len(negative2) == 0:
            continue

        # 随机选择正样本和负样本
        positive1 = positive1[torch.randint(0, len(positive1), (1,), device=rank)].unsqueeze(0)
        negative1 = negative1[torch.randint(0, len(negative1), (1,), device=rank)].unsqueeze(0)
        positive2 = positive2[torch.randint(0, len(positive2), (1,), device=rank)].unsqueeze(0)
        negative2 = negative2[torch.randint(0, len(negative2), (1,), device=rank)].unsqueeze(0)

        # 计算三元组损失
        loss_triplet1 = criterion_triplet(anchor1, positive1, negative1)
        loss_triplet2 = criterion_triplet(anchor2, positive2, negative2)
        loss_ce = F.cross_entropy(output, target)
        loss = loss_triplet1 + loss_triplet2 + loss_ce
        
        loss.backward()
        optimizer.step()

        # 累加当前 GPU 的损失值
        avg_loss += loss.item()

        # 计算当前 GPU 的预测正确数
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()

    # 计算当前 GPU 的平均损失和正确率
    avg_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)

    # 将所有 GPU 的损失值和正确数同步到主进程（rank 0）
    avg_loss_tensor = torch.tensor(avg_loss, device=rank)
    train_acc_tensor = torch.tensor(train_acc, device=rank)
    dist.reduce(avg_loss_tensor, dst=0, op=dist.ReduceOp.SUM)  # 全局求和
    dist.reduce(train_acc_tensor, dst=0, op=dist.ReduceOp.SUM)

    # 在主进程（rank 0）中计算全局平均损失和正确率
    if rank == 0:
        global_avg_loss = avg_loss_tensor.item() / dist.get_world_size()  # 平均损失
        global_train_acc = train_acc_tensor.item() / dist.get_world_size()  # 平均正确率
        print(f'Train Epoch: {epoch}, Loss: {global_avg_loss:.6f}, Acc: {global_train_acc:.4f}')
        logging.info(f'Train Epoch: {epoch}, Loss: {global_avg_loss:.6f}, Acc: {global_train_acc:.4f}')

# 验证函数
def validate(epoch, model, val_loader, rank, args):
    model.eval()
    test_loss = 0.0
    correct = 0.0
    total = 0  # 初始化 total
    accuracy = 0.0  # 设置默认值

    with torch.no_grad():
        for image1, image2, target in tqdm(val_loader, desc="Validating", disable=(rank != 0)):
            image1 = image1.to(rank)
            image2 = image2.to(rank)
            target = target.to(rank)

            # 提取模型输出的正确部分
            _, _, output = model(image1, image2)
            _, predicted = torch.max(output, 1)  # 获取预测类别
            total += target.size(0)
            correct += (predicted == target).sum().item()  # 统计预测正确的样本数

    # 将所有 GPU 的统计结果同步到主进程（rank 0）
    total_tensor = torch.tensor(total, device=rank)
    correct_tensor = torch.tensor(correct, device=rank)
    dist.reduce(total_tensor, dst=0, op=dist.ReduceOp.SUM)  # 全局求和
    dist.reduce(correct_tensor, dst=0, op=dist.ReduceOp.SUM)

    if rank == 0:
        accuracy = 100 * correct_tensor.item() / total_tensor.item()
        print(f'Test Set: Accuracy: {accuracy:.2f}%')
        logging.info(f'Test Set: Accuracy: {accuracy:.2f}%')

    return accuracy

# 主函数
def main(rank, world_size, args):
    setup(rank, world_size)

    # 数据加载
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_file_pairs = [
        ("/root/autodl-tmp/labels/original_6frame.txt", "/root/autodl-tmp/labels/originalevent_6frame.txt"),
        ("/root/autodl-tmp/labels/adacof25_30_6frame.txt","/root/autodl-tmp/labels/adacof25_30event_6frame.txt"),
    ]

    test_file_pairs = [
        ("/root/autodl-tmp/labels/rrin25_60_6frame_labels.txt", "/root/autodl-ttmp/labels/eventrrin25_60_6frame_labels.txt"),
    ]

    # 创建数据集
    full_dataset = dataset_dualstream.DualStreamDataset(train_file_pairs, transform=transform)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=10,
        pin_memory=True,
        shuffle=False  # Sampler 已经处理 shuffle
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        sampler=test_sampler,
        num_workers=10,
        pin_memory=True,
        shuffle=False
    )

    # 模型定义
    model = combine_net_event.DualCombinedModel(
        in_channels=18,
        in_channels_2=15,
        num_classes=10,
        convnext_type='base',
        convnext_type_2='small'
    ).to(rank)

    # 使用 DDP 包装模型
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # 优化器
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # 训练循环
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train(epoch, model, train_loader, optimizer, rank, args)
        val_acc = validate(epoch, model, test_loader, rank, args)

        if rank == 0 and val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存全局最佳模型
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved with accuracy: {best_val_acc:.2f}%")
            logging.info(f"New best model saved with accuracy: {best_val_acc:.2f}%")

    if rank == 0:
        print(f'Best Test Accuracy: {best_val_acc:.2f}%')

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DDP Training')
    parser.add_argument('--dataset', type=str, default='fliter')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--net-name', default='dy_resnet18')
    args = parser.parse_args()

    # 获取 GPU 数量
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs!")

    # 启动多进程训练
    torch.multiprocessing.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )