"""
统一的随机种子管理
替代各文件中散落的 torch.manual_seed / np.random.seed 调用
"""
import torch
import numpy as np


def set_seed(seed: int = 0):
    """设置所有随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
