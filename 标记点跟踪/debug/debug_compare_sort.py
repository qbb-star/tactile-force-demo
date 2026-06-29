"""
对比：旧排序（纯k-means）vs 新排序（多策略择优）
用拓扑评分量化对比改进效果
"""
import sys
sys.path.insert(0, '..')
import cv2
import numpy as np
import tac_utils
from pathlib import Path

DATA_DIR = Path(r'D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli')

test_files = [
    'image_20260430_135445_239.jpg',   # 小位移
    'image_20260430_140639_828.jpg',   # 大位移1
    'image_20260430_140728_828.jpg',   # 大位移2（最大）
    'image_20260430_140820_984.jpg',   # 大位移3
]


def old_sort_grid(keypoints):
    """旧版：纯 k-means + 贪心（Phase 1 原始版本）"""
    points = np.array(
        [(float(k.pt[0]), float(k.pt[1])) for k in keypoints],
        dtype=np.float64
    )
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    proj = centered @ R

    if np.corrcoef(proj[:, 0], centered[:, 0])[0, 1] < 0:
        proj[:, 0] *= -1
    if np.corrcoef(proj[:, 1], centered[:, 1])[0, 1] < 0:
        proj[:, 1] *= -1

    pc1_is_horizontal = abs(R[0, 0]) > abs(R[0, 1])
    if pc1_is_horizontal:
        col_base = proj[:, 0]
        row_base = proj[:, 1]
    else:
        col_base = proj[:, 1]
        row_base = proj[:, 0]

    x_centers = tac_utils._kmeans_1d(col_base, tac_utils.N_COLS)
    y_centers = tac_utils._kmeans_1d(row_base, tac_utils.N_ROWS)

    col_idx = np.argmin(np.abs(col_base[:, None] - x_centers[None, :]), axis=1)
    row_idx = np.argmin(np.abs(row_base[:, None] - y_centers[None, :]), axis=1)

    if np.corrcoef(col_idx, points[:, 0])[0, 1] < 0:
        col_idx = tac_utils.N_COLS - 1 - col_idx
        x_centers = x_centers[::-1]
    if np.corrcoef(row_idx, points[:, 1])[0, 1] < 0:
        row_idx = tac_utils.N_ROWS - 1 - row_idx
        y_centers = y_centers[::-1]

    costs = (col_base - x_centers[col_idx]) ** 2 + (row_base - y_centers[row_idx]) ** 2
    order_pts = np.argsort(costs)

    grid_points = np.full((tac_utils.N_ROWS, tac_utils.N_COLS, 2), np.nan, dtype=np.float64)
    for i in order_pts:
        r = int(row_idx[i])
        c = int(col_idx[i])
        if 0 <= r < tac_utils.N_ROWS and 0 <= c < tac_utils.N_COLS and np.isnan(grid_points[r, c, 0]):
            grid_points[r, c] = points[i]

    return grid_points


print("=" * 80)
print("排序改进前后对比：拓扑问题数量（越少越好，0=完美）")
print("=" * 80)
print(f"{'文件名':<45} {'旧版(k-means)':>14} {'新版(多策略)':>14} {'改进':>8}")
print("-" * 80)

for fname in test_files:
    img_path = DATA_DIR / fname
    img = cv2.imread(str(img_path))
    keypoints, _ = tac_utils.detect_blobs(img)

    grid_old = old_sort_grid(keypoints)
    grid_new = tac_utils.sort_grid(keypoints)

    score_old = tac_utils.grid_topology_score(grid_old)
    score_new = tac_utils.grid_topology_score(grid_new)

    valid_old = int(np.sum(~np.isnan(grid_old[:, :, 0])))
    valid_new = int(np.sum(~np.isnan(grid_new[:, :, 0])))

    improvement = score_old - score_new

    short_name = fname[:40]
    print(f"{short_name:<45} {score_old:>8}个问题  {score_new:>8}个问题  {improvement:>+6}")

    if score_old > 0:
        _, issues_old = tac_utils.validate_grid_topology(grid_old)
        print(f"  旧版问题：{issues_old[:3]}...")

print("-" * 80)
print()
print("说明：")
print("  拓扑问题 = 行/列不单调 + 间距高度不均匀")
print("  0个问题 = 网格排列完美规整")
print("  问题越多 = 排序越混乱")
