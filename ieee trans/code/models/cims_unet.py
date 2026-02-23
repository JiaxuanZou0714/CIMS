"""
CIMS-UNet1D 分割模型
Channel Independent Multi-Scale UNet: 将嵌入通道分为 4 组，
每组用独立的 UNet1D 处理不同频率特征，最后合并分类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import DataEmbedding_wo_pos_temp, PositionalEmbedding
from models.unet import UNet1D
from config import D_MODEL, NUM_CLASSES, CIMS_KERNEL_SIZES


class CIMSUnet1DSeg(nn.Module):
    """
    Channel Independent Multi-Scale UNet1D 分割模型（论文提出的方法）

    DataEmbedding_wo_pos_temp 的多尺度卷积会将 d_model 个通道分为 4 份，
    每一份对应一种 kernel size 的特征提取。
    本模型用 4 个独立的 UNet1D 分别处理这 4 份数据，最后合并输出。

    Args:
        num_classes: 分割类别数
        embed: 是否使用嵌入层
    """

    def __init__(self, num_classes=NUM_CLASSES, embed=True):
        super().__init__()
        self.embed = embed
        self.d_model = D_MODEL
        d_model = self.d_model
        self.num_classes = num_classes

        self.embedding = DataEmbedding_wo_pos_temp(1, d_model)
        self.pos_embedding = PositionalEmbedding(d_model)

        kernel_sizes = CIMS_KERNEL_SIZES
        self.unet1 = UNet1D(d_model // 4, d_model // 4, 32, kernel_sizes, kernel_sizes)
        self.unet2 = UNet1D(d_model // 4, d_model // 4, 32, kernel_sizes, kernel_sizes)
        self.unet3 = UNet1D(d_model // 4, d_model // 4, 32, kernel_sizes, kernel_sizes)
        self.unet4 = UNet1D(d_model // 4, d_model // 4, 32, kernel_sizes, kernel_sizes)

        dilation = 1
        kernel_size = 5
        padding = dilation * (kernel_size - 1) // 2

        self.classifier = nn.Sequential(
            nn.Conv1d(
                d_model, num_classes, kernel_size=kernel_size,
                padding=padding, dilation=dilation, padding_mode="circular",
            ),
        )

    def forward(self, x):
        if self.embed:
            x = self.embedding(x) + self.pos_embedding(x)

        # 将 d_model 维度分为 4 组，每组对应不同频率
        x1 = x[:, :, :4]
        x2 = x[:, :, 4:8]
        x3 = x[:, :, 8:12]
        x4 = x[:, :, 12:]

        # 独立的 UNet 处理
        x1 = self.unet1(x1)
        x2 = self.unet2(x2)
        x3 = self.unet3(x3)
        x4 = self.unet4(x4)

        # 合并 + 分类
        x = torch.cat([x1, x2, x3, x4], dim=-1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.classifier(x)
        x = x.permute(0, 2, 1).contiguous()
        x = F.softmax(x, dim=-1)
        return x
