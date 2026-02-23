"""
ResNet 分割模型
将 1D 序列嵌入为 2D 特征图，再用 ResNet-18 提取特征进行分割
"""
import torch.nn as nn
import torch.nn.functional as F

from models.embed import DataEmbedding_wo_pos_temp
from models.resnet import resnet18
from config import D_MODEL, NUM_CLASSES


class ResNetSeg(nn.Module):
    """
    基于 ResNet-18 的分割模型

    流程：Token Embedding → unsqueeze → ResNet → squeeze → Linear 分类
    """

    def __init__(self, d_model=D_MODEL, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        self.embedding = DataEmbedding_wo_pos_temp(1, d_model)
        self.resnet = resnet18(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)       # (B, 1, seq_len, d_model) — 当作单通道 2D 图像
        x = self.resnet(x)
        x = x.squeeze(1)         # (B, seq_len, d_model)
        x = self.classifier(x)
        return F.softmax(x, dim=-1)
