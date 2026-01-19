# import torch
# import torch.nn as nn
# from dynamic_conv import Dynamic_conv2d  # 假设 Dynamic_conv2d 已经定义
#
# class SimpleDynamicConvNet(nn.Module):
#     def __init__(self, in_channels=3, num_classes=1000):
#         super(SimpleDynamicConvNet, self).__init__()
#         # 动态卷积层
#         self.conv = Dynamic_conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
#         # 全局平均池化层
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # 全连接层
#         self.fc = nn.Linear(64, num_classes)
#
#     def forward(self, x):
#         # 动态卷积
#         x = self.conv(x)
#         # 全局平均池化
#         x = self.avgpool(x)
#         # 展平
#         x = torch.flatten(x, 1)
#         # 全连接层
#         x = self.fc(x)
#         return x
#
# # 实例化模型
# model = SimpleDynamicConvNet(in_channels=3, num_classes=1000)
#
# # 打印模型结构
# print(model)

import torch
import torch.nn as nn
from dynamic_conv import Dynamic_conv2d  # 假设 Dynamic_conv2d 已经定义

class SimpleDynamicConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleDynamicConvNet, self).__init__()
        # 动态卷积层
        self.conv = Dynamic_conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Dynamic_conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        # 全局平均池化层
        #self.avgpool = nn.AdaptiveAvgPool2d((5, 5))
        # 全连接层
        self.fc = nn.Linear(196608, 1000)
        self.fc1 = nn.Linear(1000, num_classes)

    def forward(self, x):
        # 动态卷积
        #print("Input to conv(x)-x layer:", x.shape)
        x1 = self.conv(x)
        x = self.conv2(x1) - x
        #print("Input to conv(x)-x layer:", x.shape)
        # 全局平均池化
        #x = self.avgpool(x)
        #print("Input to avgpool layer:", x.shape)
        # 展平
        x = torch.flatten(x, 1)
        # 打印输入到 fc 层的张量形状
        print("Input to fc layer:", x.shape)
        # 全连接层
        x = self.fc(x)
        x = self.fc1(x)
        return x
if __name__ == '__main__':
    # 实例化模型
    model = SimpleDynamicConvNet(in_channels=3, num_classes=2)

    # 创建随机输入张量（模拟一批图像数据）
    input_tensor = torch.randn(8, 3, 256, 256)  # batch_size=8, channels=3, height=224, width=224

    # 前向传播
    output = model(input_tensor)