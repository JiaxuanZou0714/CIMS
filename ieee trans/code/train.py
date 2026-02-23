"""
CIMS 地震数据分割 - 训练脚本

用法:
    python train.py
"""
import os
import gc

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import (
    SEED, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    TEST_SIZE, TRAIN_NUM_DATA, EVAL_NUM_DATA, NUM_CLASSES, DEVICE,
)
from data.dataset import GetData
from models import CIMSUnet1DSeg, ResNetSeg, TimesNetSeg
from models.unet import UNet1D
from utils.seed import set_seed
from utils.metrics import compute_segmentation_metrics

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def create_model(model_name="cims"):
    """
    创建模型

    Args:
        model_name: 模型名称，可选 'cims', 'resnet', 'timesnet', 'unet'
    """
    model_map = {
        "cims": CIMSUnet1DSeg,
        "resnet": ResNetSeg,
        "timesnet": TimesNetSeg,
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_map.keys())}")
    return model_map[model_name]()


def train(model, train_loader, device, epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """训练模型"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

    return model


def evaluate(model, device, num_data=EVAL_NUM_DATA):
    """在大规模测试集上评估模型"""
    test_data, test_label = GetData(num_data)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    metrics = compute_segmentation_metrics(model, test_loader, NUM_CLASSES, device)

    print(f"Test IoU:             [{metrics['mean_iou']:.4f}]")
    print(f"Weighted Precision:   [{metrics['weighted_precision']:.4f}]")
    print(f"Weighted Recall:      [{metrics['weighted_recall']:.4f}]")
    print(f"Weighted F1:          [{metrics['weighted_f1']:.4f}]")

    return metrics


if __name__ == "__main__":
    # 初始化
    set_seed(SEED)
    torch.cuda.empty_cache()
    print(f"Device: {DEVICE}")

    # 创建模型
    model = create_model("cims")

    # 加载训练数据
    data, label = GetData(TRAIN_NUM_DATA)
    train_data, _, train_label, _ = train_test_split(data, label, test_size=TEST_SIZE, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 训练
    model = train(model, train_loader, DEVICE)

    # 释放训练数据
    del train_data, train_label, train_dataset, train_loader, data, label
    gc.collect()
    torch.cuda.empty_cache()

    # 评估
    evaluate(model, DEVICE)
