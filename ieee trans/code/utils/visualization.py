"""
可视化工具
从 get_data.py 中提取的三维地震数据可视化函数
"""
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

matplotlib.use("Agg")


def plot_three_surfaces(data, save_path="three_surfaces.pdf", dpi=150, sample_rate=2):
    """
    简化版的三维地震数据可视化函数

    Args:
        data: 4D numpy array 或 torch.Tensor，形状 (x, y, z, 1)
        save_path: 保存路径
        dpi: 图像分辨率
        sample_rate: 采样率（越大越快但越粗糙）

    Returns:
        fig, ax: matplotlib 图形对象
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()

    x_dim, y_dim, z_dim, _ = data.shape
    print(f"数据维度: {x_dim} × {y_dim} × {z_dim}")

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    cmap = cm.seismic
    vmin, vmax = np.percentile(data, [5, 95])
    norm = plt.Normalize(vmin, vmax)

    y_indices = np.arange(0, y_dim, sample_rate)
    z_indices = np.arange(0, z_dim, sample_rate)
    x_indices = np.arange(0, x_dim, sample_rate)

    # 绘制 X 轴最大值表面 (inline)
    y, z = np.meshgrid(y_indices, z_indices, indexing="ij")
    x = np.ones_like(y) * (x_dim - 1)
    data_slice = data[-1, ::sample_rate, ::sample_rate, 0]
    ax.plot_surface(
        x, y, z, cmap=cmap, facecolors=cmap(norm(data_slice)),
        alpha=0.9, shade=False, rcount=len(y_indices), ccount=len(z_indices),
    )

    # 绘制 Y 轴最小值表面 (crossline)
    x, z = np.meshgrid(x_indices, z_indices, indexing="ij")
    y = np.zeros_like(x)
    data_slice = data[:, 0, :, 0][::sample_rate, ::sample_rate]
    ax.plot_surface(
        x, y, z, cmap=cmap, facecolors=cmap(norm(data_slice)),
        alpha=0.9, shade=False, rcount=len(x_indices), ccount=len(z_indices),
    )

    # 绘制 Z 轴最大值表面 (time/depth slice)
    x, y = np.meshgrid(x_indices, y_indices, indexing="ij")
    z = np.ones_like(x) * (z_dim - 1)
    data_slice = data[:, :, -1, 0][::sample_rate, ::sample_rate]
    ax.plot_surface(
        x, y, z, cmap=cmap, facecolors=cmap(norm(data_slice)),
        alpha=0.9, shade=False, rcount=len(x_indices), ccount=len(y_indices),
    )

    ax.view_init(elev=30, azim=-60)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, label="label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    return fig, ax
