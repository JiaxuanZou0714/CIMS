"""
Embedding 模块
包含位置编码、Token 嵌入和各种数据嵌入组合
"""
import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """正弦-余弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """多尺度卷积 Token 嵌入（4 种不同 kernel size）"""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv2 = nn.Conv1d(c_in, d_model // 4, kernel_size=9, padding=4, padding_mode="circular", bias=False)
        self.tokenConv3 = nn.Conv1d(c_in, d_model // 4, kernel_size=13, padding=6, padding_mode="circular", bias=False)
        self.tokenConv4 = nn.Conv1d(c_in, d_model // 4, kernel_size=17, padding=8, padding_mode="circular", bias=False)
        self.tokenConv5 = nn.Conv1d(c_in, d_model // 4, kernel_size=21, padding=10, padding_mode="circular", bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        x1 = self.tokenConv2(x.permute(0, 2, 1)).transpose(1, 2)
        x2 = self.tokenConv3(x.permute(0, 2, 1)).transpose(1, 2)
        x3 = self.tokenConv4(x.permute(0, 2, 1)).transpose(1, 2)
        x4 = self.tokenConv5(x.permute(0, 2, 1)).transpose(1, 2)
        return torch.cat([x1, x2, x3, x4], dim=-1)


class TokenEmbedding1(nn.Module):
    """单卷积核 Token 嵌入（kernel_size=1）"""

    def __init__(self, c_in, d_model):
        super().__init__()
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=1, padding=0, padding_mode="circular", bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    """固定的正弦-余弦嵌入（不可训练）"""

    def __init__(self, c_in, d_model):
        super().__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """时间特征嵌入（小时/星期/天/月）"""

    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super().__init__()
        minute_size, hour_size, weekday_size, day_size, month_size = 4, 24, 7, 32, 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """基于线性层的时间特征嵌入"""

    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super().__init__()
        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        self.embed = nn.Linear(freq_map[freq], d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


# ======================== 数据嵌入组合 ========================

class DataEmbedding(nn.Module):
    """完整的数据嵌入 = Token + Position + Temporal"""

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model, embed_type, freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model, embed_type, freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """数据嵌入（无位置编码）= Token + Temporal"""

    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model, embed_type, freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model, embed_type, freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    """数据嵌入（无位置和时间编码）= Token only"""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x))


class DataEmbedding_wo_temp(nn.Module):
    """数据嵌入（无时间编码）= Token + Position"""

    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x) + self.position_embedding(x))
