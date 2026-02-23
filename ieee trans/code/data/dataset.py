"""
数据加载与预处理
从 F3 地震数据集加载数据，进行归一化、裁剪、打乱等预处理
"""
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    SEISMIC_DATA_PATH, LABEL_DATA_PATH,
    INLINE_DIM, CROSSLINE_DIM, SEQ_LEN, CHANNELS,
    CROP_MARGIN, NUM_CLASSES,
)


def normalize_data(data, method="zscore"):
    """
    数据归一化

    Args:
        data: 输入数据 (samples, seq_len, channels)
        method: 'zscore' (z-score 标准化) 或 'minmax' (最小-最大归一化)

    Returns:
        归一化后的数据
    """
    if method == "zscore":
        mean = data.mean(dim=1, keepdim=True)
        std = data.std(dim=1, keepdim=True)
        return (data - mean) / (std + 1e-8)
    elif method == "minmax":
        min_vals = data.min(dim=1, keepdim=True)[0]
        max_vals = data.max(dim=1, keepdim=True)[0]
        return (data - min_vals) / (max_vals - min_vals + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def _load_raw_data(data_path, label_path):
    """加载原始数据文件"""
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        logging.error(f"Data or label file not found: {data_path}, {label_path}")
        raise FileNotFoundError(f"Data or label file not found")

    data = np.load(data_path)
    data = torch.tensor(data, dtype=torch.float32)
    data = data.reshape(data.shape[0], data.shape[2], data.shape[3])  # (N, seq_len, channels)

    label = np.load(label_path)
    label = torch.tensor(label, dtype=torch.long)
    return data, label


def GetData(num_data, one_hot=True, shuffle=True,
            data_path=SEISMIC_DATA_PATH, label_path=LABEL_DATA_PATH):
    """
    加载并预处理地震数据

    Args:
        num_data: 加载的样本数（None 表示全部）
        one_hot: 是否对标签做 one-hot 编码
        shuffle: 是否打乱数据
        data_path: 地震数据文件路径
        label_path: 标签文件路径

    Returns:
        data: (num_data, seq_len, channels)
        label: (num_data, seq_len, num_classes) if one_hot else (num_data, seq_len)
    """
    data, label = _load_raw_data(data_path, label_path)

    # reshape 为 (inline, crossline, seq_len, channels)
    data = data.reshape(INLINE_DIM, CROSSLINE_DIM, SEQ_LEN, CHANNELS)
    label = label.reshape(INLINE_DIM, CROSSLINE_DIM, SEQ_LEN)

    # 裁剪边缘
    m = CROP_MARGIN
    data = data[m:-m, m:-m, :, :]
    label = label[m:-m, m:-m, :]

    # 展平为样本列表
    data = data.reshape(-1, SEQ_LEN, CHANNELS)
    label = label.reshape(-1, SEQ_LEN)

    if shuffle:
        indices = torch.randperm(data.shape[0])
        data = data[indices]
        label = label[indices]

    if num_data is not None:
        data = data[:num_data]
        label = label[:num_data]

    data = normalize_data(data, method="zscore")

    if one_hot:
        label = F.one_hot(label, num_classes=NUM_CLASSES).float()

    label = label.float()
    return data, label


def GetSlice(n, data_path=SEISMIC_DATA_PATH, label_path=LABEL_DATA_PATH):
    """
    获取第 n 条 inline 的完整切片数据

    Args:
        n: inline 索引
        data_path: 地震数据文件路径
        label_path: 标签文件路径

    Returns:
        data: (crossline, seq_len, channels)
        label: (crossline, seq_len)
    """
    data, label = _load_raw_data(data_path, label_path)

    data = normalize_data(data, method="zscore")

    data = data.reshape(INLINE_DIM, CROSSLINE_DIM, SEQ_LEN, CHANNELS)
    label = label.reshape(INLINE_DIM, CROSSLINE_DIM, SEQ_LEN)

    m = CROP_MARGIN
    data = data[m:-m, m:-m, :, :]
    label = label[m:-m, m:-m, :]

    return data[n, :, :, :], label[n, :, :]
