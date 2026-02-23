# CIMS - Channel Independent Multi-Scale UNet for Seismic Facies Segmentation

用于 IEEE Transactions on Geoscience and Remote Sensing (TGRS) 论文的地震相分割项目。

## 项目结构

```
code/
├── config.py              # 集中配置（超参数、路径等）
├── train.py               # 训练 & 评估入口
├── data/
│   ├── __init__.py
│   └── dataset.py         # 数据加载与预处理
├── models/
│   ├── __init__.py         # 统一模型导出
│   ├── cims_unet.py        # ⭐ CIMSUnet1DSeg（论文提出方法）
│   ├── unet.py             # 1D UNet 基础网络
│   ├── embed.py            # 多尺度 Token & 位置嵌入
│   ├── conv_blocks.py      # Inception 卷积块
│   ├── resnet.py           # ResNet-18 (2D)
│   ├── resnet_seg.py       # ResNet 分割模型
│   ├── timesnet.py         # TimesNet 分割模型
│   └── informer.py         # Informer 分割模型
├── utils/
│   ├── __init__.py
│   ├── seed.py             # 随机种子管理
│   ├── metrics.py          # 分割评估指标（IoU, F1 等）
│   └── visualization.py   # 三维可视化
├── scripts/
│   └── npy_to_mat.py       # 数据格式转换工具
├── F3_seismic/             # F3 地震数据集
│   ├── F3_seismic.npy
│   └── test_label_no_ohe.npy
└── README.md
```

## 快速开始

### 训练
```bash
python train.py
```

### 切换模型
编辑 `train.py` 中的 `create_model()` 调用：
```python
model = create_model("cims")       # ⭐ 论文方法
model = create_model("resnet")     # ResNet baseline
model = create_model("timesnet")   # TimesNet baseline
```

### 修改超参数
编辑 `config.py`：
```python
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
D_MODEL = 16
```

## 模型说明

| 模型 | 文件 | 说明 |
|------|------|------|
| **CIMSUnet1DSeg** | `models/cims_unet.py` | 论文提出方法：通道独立多尺度 UNet |
| ResNetSeg | `models/resnet_seg.py` | ResNet-18 baseline |
| TimesNetSeg | `models/timesnet.py` | TimesNet baseline |
| Informer | `models/informer.py` | ProbSparse Attention baseline |
