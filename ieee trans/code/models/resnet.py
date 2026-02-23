"""
ResNet-18 (2D) 用于序列分割
将 1D 序列嵌入为 2D 特征图后，用 ResNet 提取特征
注意：此实现保留空间维度以用于分割任务（无全局平均池化）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet BasicBlock（2 层 3x3 卷积 + 残差连接）"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out, inplace=True)


class ResNet18(nn.Module):
    """
    适配分割任务的 ResNet-18
    保留序列长度维度（仅在 d_model 维度降采样后恢复），适用于逐点分类

    Args:
        in_channels: 输入通道数（通常为 1）
    """

    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=(1, 2))   # 只在 d_model 方向降采样
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=(1, 2))
        self.layer4 = self._make_layer(256, 64, blocks=2, stride=(1, 2))

        # 恢复 d_model 维度
        self.upsample = nn.ConvTranspose2d(64, 1, kernel_size=(1, 8), stride=(1, 8))

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 1, seq_len, d_model)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upsample(x)  # (B, 1, seq_len, d_model)
        return x


def resnet18(in_channels=1):
    """创建 ResNet-18 实例"""
    return ResNet18(in_channels)
