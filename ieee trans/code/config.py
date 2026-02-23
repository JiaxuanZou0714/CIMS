"""
CIMS Project Configuration
集中管理所有超参数和路径配置
"""

# ======================== 数据配置 ========================
DATA_DIR = "F3_seismic"
SEISMIC_DATA_PATH = f"{DATA_DIR}/F3_seismic.npy"
LABEL_DATA_PATH = f"{DATA_DIR}/test_label_no_ohe.npy"

# 原始数据维度 (inline × crossline × time_samples × channels)
INLINE_DIM = 601
CROSSLINE_DIM = 951
SEQ_LEN = 288
CHANNELS = 1

# 数据裁剪边界（去除边缘噪声）
CROP_MARGIN = 5

# ======================== 模型配置 ========================
NUM_CLASSES = 7
D_MODEL = 16
BASE_CHANNELS = 32

# CIMSUnet1D 专用
CIMS_KERNEL_SIZES = [7, 9, 11, 21, 31]

# TimesNet 专用
TIMESNET_D_FF = 32
TIMESNET_NUM_KERNELS = 6
TIMESNET_E_LAYERS = 3
TIMESNET_TOP_K = 2

# ======================== 训练配置 ========================
SEED = 0
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-4
TEST_SIZE = 0.5
TRAIN_NUM_DATA = 64
EVAL_NUM_DATA = 50000

# ======================== 设备配置 ========================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
