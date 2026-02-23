"""
TimesNet 分割模型
基于 FFT 提取周期性，将 1D 序列变换为 2D 后用 Inception 卷积处理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.embed import DataEmbedding_wo_pos_temp
from models.conv_blocks import Inception_Block_V2
from config import D_MODEL, NUM_CLASSES, SEQ_LEN


def FFT_for_Period(x, k=2):
    """
    通过 FFT 找到输入序列中最显著的 k 个周期

    Args:
        x: (B, T, C) 输入序列
        k: 返回的 top-k 周期数

    Returns:
        period: top-k 周期长度
        period_weight: 对应的频率幅值权重
    """
    xf = torch.fft.rfft(x, dim=1)
    # 在 batch 和 channel 维度取平均幅值
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 去除直流分量
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """TimesNet 基本块：FFT 找周期 → reshape 为 2D → Inception 卷积 → 自适应聚合"""

    def __init__(self, seq_len=SEQ_LEN, d_model=D_MODEL, d_ff=32, num_kernels=6, k=3):
        super().__init__()
        self.seq_len = seq_len
        self.k = k
        self.conv = nn.Sequential(
            Inception_Block_V2(d_model, d_ff, num_kernels=num_kernels),
            nn.ReLU(inplace=True),
            Inception_Block_V2(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # 对齐到周期整数倍长度
            if self.seq_len % period != 0:
                length = ((self.seq_len // period) + 1) * period
                padding = torch.zeros([B, length - self.seq_len, N]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len
                out = x

            # 1D → 2D：reshape 为 (B, C, num_periods, period)
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :self.seq_len, :])

        res = torch.stack(res, dim=-1)

        # 自适应加权聚合
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        return res + x  # 残差连接


class TimesNet(nn.Module):
    """
    TimesNet 编码器

    Args:
        d_model: 嵌入维度
        d_ff: 前馈层维度
        d_out: 输出维度
        num_kernels: Inception 卷积核数量
        e_layers: 编码层数
        k: FFT top-k 周期数
        embed: 是否使用嵌入层
    """

    def __init__(self, d_model=D_MODEL, d_ff=32, d_out=None, num_kernels=6, e_layers=2, k=3, embed=True):
        super().__init__()
        if d_out is None:
            d_out = d_model
        self.layer = e_layers
        self.embed = embed

        self.model = nn.ModuleList([
            TimesBlock(SEQ_LEN, d_model, d_ff, num_kernels, k) for _ in range(e_layers)
        ])
        self.enc_embedding = DataEmbedding_wo_pos_temp(1, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.last_layer = nn.Linear(d_model, d_out)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)
        if self.embed:
            enc_out = self.enc_embedding(x)
        else:
            enc_out = x
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        return enc_out


class TimesNetSeg(nn.Module):
    """TimesNet 分割模型"""

    def __init__(self, d_model=D_MODEL, d_ff=32, num_kernels=6, e_layers=3, k=2, num_classes=NUM_CLASSES):
        super().__init__()
        self.timesnet = TimesNet(d_model=d_model, d_ff=d_ff, d_out=d_model,
                                 num_kernels=num_kernels, e_layers=e_layers, k=k)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.timesnet(x)
        x = self.classifier(x)
        return F.softmax(x, dim=-1)
