"""
工具脚本：将 .npy 标签数据转换为 .mat 格式并提取 3D 边界
"""
import numpy as np
import scipy.io as sio


def extract_3d_boundary(seismic_data):
    """
    提取三维数据中的标签边界

    Args:
        seismic_data: 3D numpy array (inline, crossline, time)

    Returns:
        boundary_map: 同形状的边界标记数组 (0/1)
    """
    diff_inline = np.abs(np.diff(seismic_data, axis=0, prepend=seismic_data[0:1, :, :]))
    diff_crossline = np.abs(np.diff(seismic_data, axis=1, prepend=seismic_data[:, 0:1, :]))
    diff_time = np.abs(np.diff(seismic_data, axis=2, prepend=seismic_data[:, :, 0:1]))
    boundary_map = (diff_inline > 0) | (diff_crossline > 0) | (diff_time > 0)
    return boundary_map.astype(np.int8)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert .npy label to .mat with boundary extraction")
    parser.add_argument("input", help="Input .npy file path")
    parser.add_argument("-o", "--output", default="boundary.mat", help="Output .mat file path")
    args = parser.parse_args()

    seismic_labels = np.load(args.input)
    print(f"Input shape: {seismic_labels.shape}")

    boundary_3d = extract_3d_boundary(seismic_labels)
    boundary_indices = np.argwhere(boundary_3d == 1)

    sio.savemat(args.output, {"boundary_3d": boundary_3d, "boundary_indices": boundary_indices})
    print(f"Saved to {args.output}")
