"""
debug_big_displacement.py — 调试大位移下排序失效原因
可视化：行方向投影分布 + k-means聚类中心
"""
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tac_utils

DATA_DIR = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

# 一张小位移、一张大位移
small_file = "image_20260430_135445_239.jpg"   # max=0.19mm, 正常
big_file = "image_20260430_140728_828.jpg"      # max=1.24mm, 出错


def analyze_image(filepath, title_prefix):
    img = cv2.imread(filepath)
    keypoints, dog = tac_utils.detect_blobs(img)

    points = np.array([(k.pt[0], k.pt[1]) for k in keypoints], dtype=np.float64)

    # PCA
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    proj = centered @ R

    # 判断行/列方向
    pc1_is_horizontal = abs(R[0, 0]) > abs(R[0, 1])
    if pc1_is_horizontal:
        col_base = proj[:, 0]
        row_base = proj[:, 1]
    else:
        col_base = proj[:, 1]
        row_base = proj[:, 0]

    # k-means
    x_centers = tac_utils._kmeans_1d(col_base, tac_utils.N_COLS)
    y_centers = tac_utils._kmeans_1d(row_base, tac_utils.N_ROWS)

    return points, row_base, y_centers, col_base, x_centers


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for i, (fname, label) in enumerate([
        (small_file, "Small displacement (OK)"),
        (big_file, "Big displacement (WRONG)"),
    ]):
        fpath = os.path.join(DATA_DIR, fname)
        points, row_base, y_centers, col_base, x_centers = analyze_image(fpath, label)

        # 行方向分布
        ax = axes[i, 0]
        ax.hist(row_base, bins=60, alpha=0.6, color='steelblue', edgecolor='white')
        for yc in y_centers:
            ax.axvline(yc, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f"{label}\nRow projection distribution + k-means centers (red lines)", fontsize=11)
        ax.set_xlabel("Row coordinate (PCA projected)")
        ax.set_ylabel("Count")
        y_min, y_max = row_base.min(), row_base.max()
        ax.text(0.02, 0.95, f"{len(points)} points\n21 row centers\n"
                f"row range: {y_max-y_min:.1f}\n"
                f"avg spacing: {(y_max-y_min)/20:.1f}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 检测点散点图
        ax = axes[i, 1]
        ax.scatter(points[:, 0], points[:, 1], s=3, c='blue', alpha=0.6)
        ax.set_title(f"{label}\nDetected marker positions (raw)", fontsize=11)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.invert_yaxis()
        ax.set_aspect('equal')

    fig.suptitle(f"Why big displacement fails: k-means row assignment\n"
                 f"Small: {small_file} | Big: {big_file}", fontsize=14)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "debug_big_displacement.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {out_path}")

    # 打印统计
    print("\n=== 对比统计 ===")
    for fname, label in [(small_file, "Small"), (big_file, "Big")]:
        fpath = os.path.join(DATA_DIR, fname)
        points, row_base, y_centers, col_base, x_centers = analyze_image(fpath, label)

        # 计算行间距方差
        y_sorted = np.sort(y_centers)
        spacings = np.diff(y_sorted)
        print(f"\n{label} ({fname}):")
        print(f"  点数: {len(points)}")
        print(f"  行方向范围: {row_base.max()-row_base.min():.1f} px")
        print(f"  平均行间距: {np.mean(spacings):.2f} px")
        print(f"  行间距标准差: {np.std(spacings):.2f} px")
        print(f"  最小行间距: {np.min(spacings):.2f} px")
        print(f"  最大行间距: {np.max(spacings):.2f} px")
        print(f"  间距变异系数: {np.std(spacings)/np.mean(spacings):.3f}")

    print("\n【结论】")
    print("  大位移图像中，行间距变异系数大，说明k-means聚类中心分布不均匀，")
    print("  某些行可能被合并或分裂，导致网格排序错误。")


if __name__ == "__main__":
    main()
