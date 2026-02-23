"""
分割评估指标
从 run_seg.py 中提取，提供 IoU、Precision、Recall、F1 的计算
"""
import torch
from tqdm import tqdm


def compute_segmentation_metrics(model, data_loader, num_classes, device):
    """
    在给定数据集上计算分割模型的各项指标

    Args:
        model: 分割模型（已在 device 上）
        data_loader: 测试数据的 DataLoader
        num_classes: 类别数
        device: 计算设备

    Returns:
        dict: 包含 mean_iou, weighted_precision, weighted_recall, weighted_f1
    """
    model.eval()

    iou_list = []
    global_tp = [0] * num_classes
    global_fp = [0] * num_classes
    global_fn = [0] * num_classes

    with torch.no_grad():
        for data, label in tqdm(data_loader, desc="Evaluating"):
            data, label = data.to(device), label.to(device)
            output = model(data)
            pred = output.argmax(dim=2)
            true = label.argmax(dim=2)

            iou_sum = 0
            for cls in range(num_classes):
                pred_cls = (pred == cls)
                true_cls = (true == cls)
                intersection = (pred_cls & true_cls).sum().float()
                union = (pred_cls | true_cls).sum().float()

                if union == 0:
                    iou_cls = torch.tensor(1.0, device=device)
                else:
                    iou_cls = intersection / (union + 1e-6)
                iou_sum += iou_cls

                # 累计 TP、FP、FN
                global_tp[cls] += ((pred == cls) & (true == cls)).sum().item()
                global_fp[cls] += ((pred == cls) & (true != cls)).sum().item()
                global_fn[cls] += ((pred != cls) & (true == cls)).sum().item()

            iou_list.append((iou_sum / num_classes).item())

    # 计算加权平均指标
    total_support = 0
    weighted_precision_sum = 0
    weighted_recall_sum = 0
    weighted_f1_sum = 0

    for cls in range(num_classes):
        tp = global_tp[cls]
        fp = global_fp[cls]
        fn = global_fn[cls]
        support = tp + fn
        total_support += support

        precision_cls = tp / (tp + fp + 1e-6)
        recall_cls = tp / (tp + fn + 1e-6)
        f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-6)

        weighted_precision_sum += precision_cls * support
        weighted_recall_sum += recall_cls * support
        weighted_f1_sum += f1_cls * support

    metrics = {
        "mean_iou": sum(iou_list) / len(iou_list),
        "weighted_precision": weighted_precision_sum / (total_support + 1e-6),
        "weighted_recall": weighted_recall_sum / (total_support + 1e-6),
        "weighted_f1": weighted_f1_sum / (total_support + 1e-6),
    }
    return metrics
