import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

INLINE_DIM = 601
CROSSLINE_DIM = 951
CROP_MARGIN = 5

valid_inline = INLINE_DIM - 2 * CROP_MARGIN
valid_crossline = CROSSLINE_DIM - 2 * CROP_MARGIN
total_samples = valid_inline * valid_crossline

# Create coordinate grid
inline_indices = np.arange(CROP_MARGIN, INLINE_DIM - CROP_MARGIN)
crossline_indices = np.arange(CROP_MARGIN, CROSSLINE_DIM - CROP_MARGIN)

coords = []
for i in inline_indices:
    for x in crossline_indices:
        coords.append((i, x))
coords = np.array(coords)

set_seed(0)

indices = torch.randperm(total_samples)
coords = coords[indices]

num_data = 64
coords = coords[:num_data]

train_coords, val_coords = train_test_split(coords, test_size=0.5, random_state=42)

print("Train coords:", train_coords)
print("Val coords:", val_coords)

plt.figure(figsize=(10, 8))
plt.scatter(train_coords[:, 1], train_coords[:, 0], c='red', marker='o', s=50, label='Training Samples')
plt.scatter(val_coords[:, 1], val_coords[:, 0], c='blue', marker='x', s=50, label='Validation Samples')

# Draw margin area
plt.plot([CROP_MARGIN, CROSSLINE_DIM-CROP_MARGIN], [CROP_MARGIN, CROP_MARGIN], 'k--', label='Valid Area Boundary')
plt.plot([CROP_MARGIN, CROSSLINE_DIM-CROP_MARGIN], [INLINE_DIM-CROP_MARGIN, INLINE_DIM-CROP_MARGIN], 'k--')
plt.plot([CROP_MARGIN, CROP_MARGIN], [CROP_MARGIN, INLINE_DIM-CROP_MARGIN], 'k--')
plt.plot([CROSSLINE_DIM-CROP_MARGIN, CROSSLINE_DIM-CROP_MARGIN], [CROP_MARGIN, INLINE_DIM-CROP_MARGIN], 'k--')

plt.xlim(0, CROSSLINE_DIM)
plt.ylim(0, INLINE_DIM)
plt.gca().invert_yaxis()
plt.xlabel("Crossline Index")
plt.ylabel("Inline Index")
plt.title("Spatial Distribution of Randomly Sampled Traces")
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.6)

output_path = '/Users/jiaxuanzou/CIMS/ieee trans/code/spatial_distribution.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {output_path}")
