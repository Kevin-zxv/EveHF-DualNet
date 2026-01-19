import torch
import torch.nn as nn
from dynamic_conv import Dynamic_conv2d  # 假设 Dynamic_conv2d 已经定义

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3):
        super(FeatureExtractor, self).__init__()
        # 动态卷积层
        self.conv = Dynamic_conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        #self.conv2 = Dynamic_conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        # 全局平均池化层（如果需要）
        # self.avgpool = nn.AdaptiveAvgPool2d((5, 5))

    def forward(self, x):
        # 动态卷积
        x1 = self.conv(x)
        x = x - x1
        #x = x-self.conv2(x1)
        # 全局平均池化（如果需要）
        # x = self.avgpool(x)
        ############print(x)
        return x

# 分类器
class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        # 全连接层
        self.fc = nn.Linear(196608, 1000)
        self.fc1 = nn.Linear(1000, num_classes)

    def forward(self, x):
        # 展平
        x = torch.flatten(x, 1)
        # 全连接层
        x = self.fc(x)
        x = self.fc1(x)
        return x

# 完整模型（特征提取器 + 分类器）
class SimpleDynamicConvNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(SimpleDynamicConvNet, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels)
        self.classifier = Classifier(num_classes)

    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)
        # 分类
        output = self.classifier(features)
        return output

if __name__ == '__main__':
    # 实例化模型
    model = SimpleDynamicConvNet(in_channels=3, num_classes=2)

    # 创建随机输入张量（模拟一批图像数据）
    input_tensor = torch.randn(8, 3, 256, 256)  # batch_size=8, channels=3, height=256, width=256

    # 前向传播
    output = model(input_tensor)
    print("Output shape:", output.shape)  # 输出形状应为 (batch_size, num_classes)