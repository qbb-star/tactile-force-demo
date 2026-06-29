"""
demo_04_homography.py — 透视矫正（Homography）
==============================================
目的：理解透视变换矩阵如何将倾斜的网格矫正为"鸟瞰图"。

算法原理：
  透视变换（Perspective Transform）是一种 3×3 的变换矩阵，
  能将一个四边形映射到另一个四边形。

  在我们的场景中：
  - 源四边形：四个角点在原始图像中的位置（通常是不规则四边形）
  - 目标四边形：四个角点在标准网格中的位置（矩形）

  数学表达：
    [x']   [h11 h12 h13] [x]
    [y'] = [h21 h22 h23] [y]
    [w']   [h31 h32  1 ] [1]

    实际坐标：x" = x'/w', y" = y'/w'

  8 个未知数（h11-h32），需要 4 对对应点来求解。
  OpenCV 的 getPerspectiveTransform 就是做这件事。

  局限性：
  - 单一 Homography 只能矫正透视，不能矫正镜头畸变
  - 对于非线性畸变，需要逐网格单元做仿射变换（见 demo_05）

运行：python demo_04_homography.py
"""

import cv2
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os

# ============================================================
# 配置
# ============================================================
DATA_DIR = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
IMAGE_FILE = "ref.jpg"  # 无接触参考帧
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_COLS = 16
N_ROWS = 21
N_EXPECTED = N_COLS * N_ROWS

ROI_X0, ROI_Y0 = 810, 225
ROI_X1, ROI_Y1 = 1140, 630

# ============================================================
# 1. 复用 demo_03 的检测+排序逻辑
# ============================================================
img_path = os.path.join(DATA_DIR, IMAGE_FILE)
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# DoG + Blob 检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
blur_large = cv2.GaussianBlur(gray, (27, 27), 0.3 * ((27 - 1) * 0.5 - 1) + 0.8)
blur_small = cv2.GaussianBlur(gray, (5, 5), 0.3 * ((5 - 1) * 0.5 - 1) + 0.8)
dog = np.clip((blur_large - blur_small) * 12.9, 0.0, 255.0).astype(np.uint8)

roi_mask = np.zeros_like(dog)
roi_mask[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1] = 255
dog = cv2.bitwise_and(dog, roi_mask)

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 171.0
params.maxThreshold = 255.0
params.thresholdStep = 11.0
params.minDistBetweenBlobs = 6.0
params.filterByArea = True
params.minArea = 8.0
params.maxArea = 1500.0
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = False
params.filterByInertia = False
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(dog)
keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)[:N_EXPECTED]
points = np.array([(float(k.pt[0]), float(k.pt[1])) for k in keypoints], dtype=np.float64)

# PCA 排序
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

# 判断哪个主成分是水平方向（x分量大）→ 列方向
pc1_is_horizontal = abs(R[0, 0]) > abs(R[0, 1])
if pc1_is_horizontal:
    col_base = proj[:, 0]; row_base = proj[:, 1]
else:
    col_base = proj[:, 1]; row_base = proj[:, 0]


def kmeans_1d(values, k, iters=50):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centers = np.linspace(values.min(), values.max(), k)
    for _ in range(iters):
        dists = np.abs(values[:, None] - centers[None, :])
        labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for i in range(k):
            cluster = values[labels == i]
            if cluster.size > 0:
                new_centers[i] = np.median(cluster)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return np.sort(centers)


x_centers = kmeans_1d(col_base, N_COLS)
y_centers = kmeans_1d(row_base, N_ROWS)

col_idx = np.argmin(np.abs(col_base[:, None] - x_centers[None, :]), axis=1)
row_idx = np.argmin(np.abs(row_base[:, None] - y_centers[None, :]), axis=1)

if np.corrcoef(col_idx, points[:, 0])[0, 1] < 0:
    col_idx = N_COLS - 1 - col_idx
    x_centers = x_centers[::-1]
if np.corrcoef(row_idx, points[:, 1])[0, 1] < 0:
    row_idx = N_ROWS - 1 - row_idx
    y_centers = y_centers[::-1]

costs = (col_base - x_centers[col_idx]) ** 2 + (row_base - y_centers[row_idx]) ** 2
order_pts = np.argsort(costs)

grid_points = np.full((N_ROWS, N_COLS, 2), np.nan, dtype=np.float64)
for i in order_pts:
    r = int(row_idx[i])
    c = int(col_idx[i])
    if 0 <= r < N_ROWS and 0 <= c < N_COLS and np.isnan(grid_points[r, c, 0]):
        grid_points[r, c] = points[i]

print("=" * 60)
print("【透视矫正 — Homography 演示】")

# ============================================================
# 2. 提取四个角点
# ============================================================
# 角点顺序：TL(左上), TR(右上), BR(右下), BL(左下)
tl = grid_points[0, 0]       # 左上角
tr = grid_points[0, -1]      # 右上角
br = grid_points[-1, -1]     # 右下角
bl = grid_points[-1, 0]      # 左下角

corners_src = np.float32([tl, tr, br, bl])

print(f"\n【源角点（原始图像坐标）】")
print(f"  TL (左上): ({tl[0]:.1f}, {tl[1]:.1f})")
print(f"  TR (右上): ({tr[0]:.1f}, {tr[1]:.1f})")
print(f"  BR (右下): ({br[0]:.1f}, {br[1]:.1f})")
print(f"  BL (左下): ({bl[0]:.1f}, {bl[1]:.1f})")

# ============================================================
# 3. 定义目标四边形（标准矩形）
# ============================================================
# 目标：在矫正图像中，网格占据一个规则的矩形区域
# 宽度 = (N_COLS - 1) * 像素间距，高度 = (N_ROWS - 1) * 像素间距
margin = 50  # 边距
pixel_spacing = 20  # 每个网格单元在矫正图像中的像素间距
rect_w = (N_COLS - 1) * pixel_spacing + 2 * margin
rect_h = (N_ROWS - 1) * pixel_spacing + 2 * margin

corners_dst = np.float32([
    [margin, margin],                          # TL
    [margin + (N_COLS - 1) * pixel_spacing, margin],  # TR
    [margin + (N_COLS - 1) * pixel_spacing, margin + (N_ROWS - 1) * pixel_spacing],  # BR
    [margin, margin + (N_ROWS - 1) * pixel_spacing],  # BL
])

print(f"\n【目标角点（矫正图像坐标）】")
print(f"  TL: ({corners_dst[0][0]:.0f}, {corners_dst[0][1]:.0f})")
print(f"  TR: ({corners_dst[1][0]:.0f}, {corners_dst[1][1]:.0f})")
print(f"  BR: ({corners_dst[2][0]:.0f}, {corners_dst[2][1]:.0f})")
print(f"  BL: ({corners_dst[3][0]:.0f}, {corners_dst[3][1]:.0f})")

# ============================================================
# 4. 计算透视变换矩阵
# ============================================================
print("\n【计算透视变换矩阵 H】")
H = cv2.getPerspectiveTransform(corners_src, corners_dst)
print(f"  H (3×3):\n{np.round(H, 4)}")

# 应用透视变换
rect_w_int = int(rect_w)
rect_h_int = int(rect_h)
warped = cv2.warpPerspective(img, H, (rect_w_int, rect_h_int))
warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

# 同时变换网格点坐标
grid_flat = grid_points.reshape(-1, 2)
valid_mask = ~np.isnan(grid_flat[:, 0])
grid_valid = grid_flat[valid_mask]
grid_homo = cv2.perspectiveTransform(
    grid_valid.reshape(-1, 1, 2).astype(np.float32), H
).reshape(-1, 2)

print(f"  矫正后图像尺寸: {rect_w_int} × {rect_h_int}")

# ============================================================
# 5. 分析矫正效果
# ============================================================
print("\n【矫正效果分析】")

# 重新排列矫正后的网格点
grid_rectified = np.full((N_ROWS, N_COLS, 2), np.nan, dtype=np.float64)
idx = 0
for r in range(N_ROWS):
    for c in range(N_COLS):
        if valid_mask[r * N_COLS + c]:
            grid_rectified[r, c] = grid_homo[idx]
            idx += 1

# 计算矫正后的行/列间距
row_spacings_rect = []
for r in range(N_ROWS - 1):
    for c in range(N_COLS):
        if not np.isnan(grid_rectified[r, c, 0]) and not np.isnan(grid_rectified[r + 1, c, 0]):
            d = np.linalg.norm(grid_rectified[r + 1, c] - grid_rectified[r, c])
            row_spacings_rect.append(d)

col_spacings_rect = []
for c in range(N_COLS - 1):
    for r in range(N_ROWS):
        if not np.isnan(grid_rectified[r, c, 0]) and not np.isnan(grid_rectified[r, c + 1, 0]):
            d = np.linalg.norm(grid_rectified[r, c + 1] - grid_rectified[r, c])
            col_spacings_rect.append(d)

print(f"  矫正后行间距: mean={np.mean(row_spacings_rect):.1f}, std={np.std(row_spacings_rect):.1f}")
print(f"  矫正后列间距: mean={np.mean(col_spacings_rect):.1f}, std={np.std(col_spacings_rect):.1f}")
print(f"  理想间距: {pixel_spacing}")

# 原始间距（用于对比）
row_spacings_raw = []
for r in range(N_ROWS - 1):
    for c in range(N_COLS):
        if not np.isnan(grid_points[r, c, 0]) and not np.isnan(grid_points[r + 1, c, 0]):
            d = np.linalg.norm(grid_points[r + 1, c] - grid_points[r, c])
            row_spacings_raw.append(d)

print(f"\n  原始行间距: mean={np.mean(row_spacings_raw):.1f}, std={np.std(row_spacings_raw):.1f}")
print(f"  → 矫正后 std 显著减小，说明网格更均匀")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 原始图像 + 角点标注
axes[0, 0].imshow(img_rgb)
for i, (name, pt) in enumerate(zip(["TL", "TR", "BR", "BL"], corners_src)):
    axes[0, 0].plot(pt[0], pt[1], "ro", markersize=8)
    axes[0, 0].text(pt[0] + 10, pt[1] - 10, name, color="red", fontsize=10, fontweight="bold")
# 绘制四边形
pts = np.vstack([corners_src, corners_src[0]])
axes[0, 0].plot(pts[:, 0], pts[:, 1], "r-", linewidth=2)
axes[0, 0].set_title("Original Image with Corner Points", fontsize=11)
axes[0, 0].axis("off")

# 矫正后的图像 + 角点
axes[0, 1].imshow(warped_rgb)
for i, (name, pt) in enumerate(zip(["TL", "TR", "BR", "BL"], corners_dst)):
    axes[0, 1].plot(pt[0], pt[1], "go", markersize=8)
    axes[0, 1].text(pt[0] + 5, pt[1] - 5, name, color="lime", fontsize=10, fontweight="bold")
pts_dst = np.vstack([corners_dst, corners_dst[0]])
axes[0, 1].plot(pts_dst[:, 0], pts_dst[:, 1], "g-", linewidth=2)
axes[0, 1].set_title("Rectified Image (Homography)", fontsize=11)
axes[0, 1].axis("off")

# 矫正后图像 + 网格 overlay
axes[0, 2].imshow(warped_rgb)
for r in range(N_ROWS):
    for c in range(N_COLS - 1):
        if not np.isnan(grid_rectified[r, c, 0]) and not np.isnan(grid_rectified[r, c + 1, 0]):
            p0 = grid_rectified[r, c]
            p1 = grid_rectified[r, c + 1]
            axes[0, 2].plot([p0[0], p1[0]], [p0[1], p1[1]], "c-", linewidth=0.5)
for c in range(N_COLS):
    for r in range(N_ROWS - 1):
        if not np.isnan(grid_rectified[r, c, 0]) and not np.isnan(grid_rectified[r + 1, c, 0]):
            p0 = grid_rectified[r, c]
            p1 = grid_rectified[r + 1, c]
            axes[0, 2].plot([p0[0], p1[0]], [p0[1], p1[1]], "orange", linewidth=0.5)
axes[0, 2].set_title("Rectified + Grid Overlay", fontsize=11)
axes[0, 2].axis("off")

# 原始行间距 vs 矫正后行间距
axes[1, 0].plot(row_spacings_raw, "b.-", alpha=0.7, label="Original")
axes[1, 0].plot(row_spacings_rect, "r.-", alpha=0.7, label="Rectified")
axes[1, 0].axhline(y=pixel_spacing, color="g", linestyle="--", label=f"Ideal ({pixel_spacing})")
axes[1, 0].set_title("Row Spacing: Original vs Rectified", fontsize=11)
axes[1, 0].set_xlabel("Row pair index")
axes[1, 0].set_ylabel("Distance (px)")
axes[1, 0].legend()

# 原始列间距 vs 矫正后列间距
col_spacings_raw = []
for c in range(N_COLS - 1):
    for r in range(N_ROWS):
        if not np.isnan(grid_points[r, c, 0]) and not np.isnan(grid_points[r, c + 1, 0]):
            d = np.linalg.norm(grid_points[r, c + 1] - grid_points[r, c])
            col_spacings_raw.append(d)

axes[1, 1].plot(col_spacings_raw, "b.-", alpha=0.7, label="Original")
axes[1, 1].plot(col_spacings_rect, "r.-", alpha=0.7, label="Rectified")
axes[1, 1].axhline(y=pixel_spacing, color="g", linestyle="--", label=f"Ideal ({pixel_spacing})")
axes[1, 1].set_title("Column Spacing: Original vs Rectified", fontsize=11)
axes[1, 1].set_xlabel("Column pair index")
axes[1, 1].set_ylabel("Distance (px)")
axes[1, 1].legend()

# 矫正前后对比（并排）
axes[1, 2].imshow(img_rgb)
axes[1, 2].set_title("Original (perspective distorted)", fontsize=11)
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "demo_04_homography.png"), dpi=150)
plt.close()
print(f"\n[保存] 可视化结果已保存到 {OUTPUT_DIR}/demo_04_homography.png")

# 同时保存矫正后的图像
cv2.imwrite(os.path.join(OUTPUT_DIR, "demo_04_rectified.jpg"), warped)
print(f"[保存] 矫正后图像已保存到 {OUTPUT_DIR}/demo_04_rectified.jpg")

# ============================================================
# 7. 总结
# ============================================================
print("\n" + "=" * 60)
print("【关键理解】")
print("  1. Homography 是 3×3 矩阵，8 个自由度，需要 4 对对应点")
print("  2. 它只能矫正透视变形（线性），不能矫正镜头畸变（非线性）")
print("  3. 矫正后网格间距更均匀，但仍有残留的非线性畸变")
print("  4. 对于高精度触觉重建，需要逐网格单元做仿射映射（见 demo_05）")
print("\n✅ demo_04 完成！")
print("  下一步: demo_05_warp.py — 学习完整 warp 构建")