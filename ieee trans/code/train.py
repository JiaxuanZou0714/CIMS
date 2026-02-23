"""
CIMS 地震数据分割 - 训练脚本

用法:
    python train.py
"""
import os
import gc

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

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


def train(model, train_loader, val_loader, device, model_name="cims",
          epochs=EPOCHS, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """训练模型，记录训练/验证损失并保存最佳模型"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in tqdm(range(epochs), desc="Training"):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        num_batches = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        avg_train_loss = running_loss / num_batches
        train_losses.append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 每 50 个 epoch 打印一次
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"Train Loss: {avg_train_loss:.4f}  "
                  f"Val Loss: {avg_val_loss:.4f}  "
                  f"Best: Epoch {best_epoch} ({best_val_loss:.4f})")

    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}.")

    # --- 绘制收敛曲线 ---
    plot_convergence(train_losses, val_losses, best_epoch, model_name)

    return model, {"train_losses": train_losses, "val_losses": val_losses,
                   "best_epoch": best_epoch, "best_val_loss": best_val_loss}


def plot_convergence(train_losses, val_losses, best_epoch, model_name="model"):
    """绘制训练/验证损失曲线并标注最佳 epoch"""
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7,
                label=f"Best Epoch ({best_epoch})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name.upper()} Convergence Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = f"convergence_{model_name}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Convergence curve saved to {save_path}")


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
    model_name = "cims"
    model = create_model(model_name)

    # 加载训练数据并划分训练/验证集
    data, label = GetData(TRAIN_NUM_DATA)
    train_data, val_data, train_label, val_label = train_test_split(
        data, label, test_size=TEST_SIZE, random_state=42
    )

    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_data, val_label)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 训练
    model, history = train(model, train_loader, val_loader, DEVICE, model_name=model_name)

    print(f"\n{'='*50}")
    print(f"Model: {model_name.upper()}")
    print(f"Converged at epoch: {history['best_epoch']} / {EPOCHS}")
    print(f"Best validation loss: {history['best_val_loss']:.4f}")
    print(f"{'='*50}\n")

    # 释放训练数据
    del train_data, val_data, train_label, val_label
    del train_dataset, val_dataset, train_loader, val_loader, data, label
    gc.collect()
    torch.cuda.empty_cache()

    # 评估
    evaluate(model, DEVICE)
