"""
1D UNet 模块
用于一维序列的编码-解码分割网络
"""
import torch
import torch.nn as nn


class DoubleConv1D(nn.Module):
    """两层 1D 卷积 + BatchNorm + ReLU"""

    def __init__(self, in_channels, out_channels, k1=5, k2=7):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k1, padding=(k1 - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=k2, padding=(k2 - 1) // 2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                m.weight.data.mul_(1.0)

    def forward(self, x):
        return self.double_conv(x)


class UNet1D(nn.Module):
    """
    1D UNet：4 层编码-解码 + 跳跃连接

    Args:
        in_channels: 输入通道数
        out_channel: 输出通道数
        base_channels: 基础通道数（逐层翻倍）
        down_kernel_size_lists: 编码器各层卷积核大小
        up_kernel_size_lists: 解码器各层卷积核大小
    """

    def __init__(self, in_channels=1, out_channel=7, base_channels=32,
                 down_kernel_size_lists=None, up_kernel_size_lists=None):
        super().__init__()
        if down_kernel_size_lists is None:
            down_kernel_size_lists = [5, 7, 9, 11, 13]
        if up_kernel_size_lists is None:
            up_kernel_size_lists = [5, 7, 9, 11, 13]

        # 编码器（下采样）
        self.inc = DoubleConv1D(in_channels, base_channels)
        self.down1 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(base_channels, base_channels * 2, down_kernel_size_lists[0], down_kernel_size_lists[1]),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(base_channels * 2, base_channels * 4, down_kernel_size_lists[1], down_kernel_size_lists[2]),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(base_channels * 4, base_channels * 8, down_kernel_size_lists[2], down_kernel_size_lists[3]),
        )
        self.down4 = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(base_channels * 8, base_channels * 16, down_kernel_size_lists[3], down_kernel_size_lists[4]),
        )

        # 解码器（上采样）
        self.up1 = nn.ConvTranspose1d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.conv1 = DoubleConv1D(base_channels * 16, base_channels * 8, up_kernel_size_lists[0], up_kernel_size_lists[1])

        self.up2 = nn.ConvTranspose1d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.conv2 = DoubleConv1D(base_channels * 8, base_channels * 4, up_kernel_size_lists[1], up_kernel_size_lists[2])

        self.up3 = nn.ConvTranspose1d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.conv3 = DoubleConv1D(base_channels * 4, base_channels * 2, up_kernel_size_lists[2], up_kernel_size_lists[3])

        self.up4 = nn.ConvTranspose1d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.conv4 = DoubleConv1D(base_channels * 2, base_channels, up_kernel_size_lists[3], up_kernel_size_lists[4])

        self.outc = nn.Conv1d(base_channels, out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        # 输入: (B, seq_len, C) → 转为 (B, C, seq_len)
        x = x.permute(0, 2, 1)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)

        logits = self.outc(x)
        return logits.permute(0, 2, 1)  # (B, seq_len, out_channel)
