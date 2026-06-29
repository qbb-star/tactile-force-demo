"""
demo_05_warp.py — 完整 warp 构建（逐三角剖分仿射映射）
=====================================================
目的：理解 viewtacv2 中使用的完整几何矫正算法。

算法原理：
  单一 Homography 只能矫正透视，无法矫正镜头畸变（非线性）。
  
  viewtacv2 使用更精细的方法：逐网格单元做仿射映射。

  流程：
  1. 检测并排序网格点（复用 demo_02 + demo_03）
  2. 边界外推：在网格外围扩展一圈虚拟点
  3. 生成标准网格（canonical grid）：保持物理纵横比
  4. 每个四边形单元拆分为两个三角形
  5. 对每个三角形计算仿射变换 (3 点 → 3 点)
  6. 遍历标准网格的每个像素，通过所属三角形的仿射变换找到原始图像中的对应位置
  7. 生成稠密的 map_x, map_y 查找表（即 warp）
  8. 保存 warp 为 npz 文件，后续可直接加载使用

  与 Homography 的区别：
  - Homography: 全局 1 个 3×3 矩阵，线性矫正
  - Warp: 逐网格单元局部仿射，能处理非线性畸变

运行：python demo_05_warp.py
"""

import cv2
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os
import json

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
PITCH_X_MM = 0.6  # 物理列间距
PITCH_Y_MM = 0.6  # 物理行间距
OUT_WIDTH = 320   # 矫正图像宽度
PAD_PX = 20.0     # 边界填充像素

ROI_X0, ROI_Y0 = 810, 225
ROI_X1, ROI_Y1 = 1140, 630

# 底部遮挡处理
# 旧图像（calibData）：左侧有遮挡（mask_left_px=400）
# 新图像（data_wenli）：底部有遮挡，ref.jpg 检测到约 320 个点（16×20）
# 底部遮挡导致最后一行标记点不可见，需要在矫正后裁剪底部
MASK_BOTTOM_PX = 20  # 矫正后图像底部裁剪像素数

# ============================================================
# Step 1: 检测并排序网格点（复用 demo_02 + demo_03 逻辑）
# ============================================================
print("=" * 60)
print("【完整 Warp 构建 — 逐步演示】")

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

print(f"  Step 1: 检测并排序完成，网格 {N_ROWS}×{N_COLS}")

# ============================================================
# Step 2: 计算原始网格间距
# ============================================================
print("\n【Step 2: 计算原始网格间距】")

# 使用 nanmedian 处理空单元格的 NaN
dx_raw_px = float(np.nanmedian(np.linalg.norm(
    grid_points[:, 1:, :] - grid_points[:, :-1, :], axis=2
)))
dy_raw_px = float(np.nanmedian(np.linalg.norm(
    grid_points[1:, :, :] - grid_points[:-1, :, :], axis=2
)))

# 如果全是 NaN，使用绝对坐标范围除以网格数估算
if np.isnan(dx_raw_px):
    valid_x = grid_points[:, :, 0][~np.isnan(grid_points[:, :, 0])]
    dx_raw_px = (valid_x.max() - valid_x.min()) / (N_COLS - 1)
if np.isnan(dy_raw_px):
    valid_y = grid_points[:, :, 1][~np.isnan(grid_points[:, :, 1])]
    dy_raw_px = (valid_y.max() - valid_y.min()) / (N_ROWS - 1)

print(f"  原始列间距中位数: dx_raw_px = {dx_raw_px:.2f} px")
print(f"  原始行间距中位数: dy_raw_px = {dy_raw_px:.2f} px")

# ============================================================
# Step 3: 边界外推（扩展网格）
# ============================================================
print("\n【Step 3: 边界外推】")
print("  在网格外围扩展一圈虚拟点，确保矫正图像包含完整视场")


def extend_grid_boundary(grid_points_raw, pad_raw_x, pad_raw_y):
    """在网格外围扩展一圈"""
    gp = np.asarray(grid_points_raw, dtype=np.float64)
    rows, cols, _ = gp.shape

    # 对每行左右扩展
    row_ext = []
    for r in range(rows):
        row = gp[r]
        # 左扩展：沿第一段方向
        left_dir = row[1] - row[0]
        left_norm = np.linalg.norm(left_dir)
        if left_norm > 1e-9:
            left = row[0] - left_dir / left_norm * pad_raw_x
        else:
            left = row[0] - np.array([pad_raw_x, 0])
        # 右扩展：沿最后一段方向
        right_dir = row[-1] - row[-2]
        right_norm = np.linalg.norm(right_dir)
        if right_norm > 1e-9:
            right = row[-1] + right_dir / right_norm * pad_raw_x
        else:
            right = row[-1] + np.array([pad_raw_x, 0])
        row_ext.append(np.vstack([left, row, right]))
    gp_lr = np.stack(row_ext, axis=0)  # (rows, cols+2, 2)

    # 对每列上下扩展
    col_ext = []
    for c in range(cols + 2):
        col = gp_lr[:, c]
        top_dir = col[1] - col[0]
        top_norm = np.linalg.norm(top_dir)
        if top_norm > 1e-9:
            top = col[0] - top_dir / top_norm * pad_raw_y
        else:
            top = col[0] - np.array([0, pad_raw_y])
        bot_dir = col[-1] - col[-2]
        bot_norm = np.linalg.norm(bot_dir)
        if bot_norm > 1e-9:
            bot = col[-1] + bot_dir / bot_norm * pad_raw_y
        else:
            bot = col[-1] + np.array([0, pad_raw_y])
        col_ext.append(np.vstack([top, col, bot]))
    return np.stack(col_ext, axis=1)  # (rows+2, cols+2, 2)


raw_ext = extend_grid_boundary(grid_points, dx_raw_px * 0.5, dy_raw_px * 0.5)
print(f"  扩展后网格: {raw_ext.shape[0]}×{raw_ext.shape[1]} (原始 {N_ROWS}×{N_COLS})")

# ============================================================
# Step 4: 生成标准网格（canonical grid）
# ============================================================
print("\n【Step 4: 生成标准网格（保持物理纵横比）】")

# 物理尺寸
phys_w = (N_COLS - 1) * PITCH_X_MM
phys_h = (N_ROWS - 1) * PITCH_Y_MM
aspect = phys_h / phys_w

print(f"  物理尺寸: {phys_w:.1f}mm × {phys_h:.1f}mm, 纵横比={aspect:.3f}")

# 矫正图像中的 ROI 宽度
roi_rect_w = OUT_WIDTH - 2 * PAD_PX
dx_rect = roi_rect_w / (N_COLS - 1)
dy_rect = dx_rect * aspect  # 保持物理纵横比

out_h_float = 2 * PAD_PX + (N_ROWS - 1) * dy_rect
out_h = int(np.ceil(out_h_float))

scale_x = dx_rect / dx_raw_px
scale_y = dy_rect / dy_raw_px

print(f"  矫正图像尺寸: {OUT_WIDTH}×{out_h}")
print(f"  矫正后列间距: dx_rect = {dx_rect:.2f} px")
print(f"  矫正后行间距: dy_rect = {dy_rect:.2f} px")
print(f"  缩放因子: scale_x = {scale_x:.4f}, scale_y = {scale_y:.4f}")

# 生成标准网格（包含扩展的边界）
xs = np.concatenate([
    [0.0],
    PAD_PX + np.arange(N_COLS, dtype=np.float64) * dx_rect,
    [float(OUT_WIDTH)]
])
ys = np.concatenate([
    [0.0],
    PAD_PX + np.arange(N_ROWS, dtype=np.float64) * dy_rect,
    [float(out_h)]
])

can_pts = np.zeros((N_ROWS + 2, N_COLS + 2, 2), dtype=np.float64)
for r in range(N_ROWS + 2):
    for c in range(N_COLS + 2):
        can_pts[r, c] = (xs[c], ys[r])

print(f"  标准网格: {can_pts.shape[0]}×{can_pts.shape[1]}")

# ============================================================
# Step 5: 逐三角形仿射映射
# ============================================================
print("\n【Step 5: 逐三角形仿射映射】")
print("  每个四边形 → 2 个三角形 → 仿射变换 → 填充查找表")


def point_in_triangle(px, py, tri):
    """判断点是否在三角形内（重心坐标法）"""
    x0, y0 = tri[0]
    x1, y1 = tri[1]
    x2, y2 = tri[2]

    v0x, v0y = x2 - x0, y2 - y0
    v1x, v1y = x1 - x0, y1 - y0
    v2x, v2y = px - x0, py - y0

    den = v0x * v1y - v1x * v0y
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)

    a = (v2x * v1y - v1x * v2y) / den
    b = (v0x * v2y - v2x * v0y) / den
    c = 1.0 - a - b
    return (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)


def dense_map_from_grids(raw_pts, can_pts, out_h, out_w):
    """从原始网格+标准网格生成稠密映射表"""
    raw_pts = np.asarray(raw_pts, dtype=np.float32)
    can_pts = np.asarray(can_pts, dtype=np.float32)

    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)

    rows, cols, _ = can_pts.shape
    cell_count = 0

    for r in range(rows - 1):
        for c in range(cols - 1):
            # 四个顶点
            q00 = can_pts[r, c]
            q10 = can_pts[r, c + 1]
            q01 = can_pts[r + 1, c]
            q11 = can_pts[r + 1, c + 1]

            p00 = raw_pts[r, c]
            p10 = raw_pts[r, c + 1]
            p01 = raw_pts[r + 1, c]
            p11 = raw_pts[r + 1, c + 1]

            # 两个三角形
            tri_q = [(q00, q10, q11), (q00, q11, q01)]
            tri_p = [(p00, p10, p11), (p00, p11, p01)]

            for tq, tp in zip(tri_q, tri_p):
                tq = np.asarray(tq, dtype=np.float32)
                tp = np.asarray(tp, dtype=np.float32)

                # 仿射变换：标准三角形 → 原始三角形
                A = cv2.getAffineTransform(tq, tp)

                # 三角形包围盒
                x_min = int(max(0, np.floor(np.min(tq[:, 0]))))
                x_max = int(min(out_w - 1, np.ceil(np.max(tq[:, 0]))))
                y_min = int(max(0, np.floor(np.min(tq[:, 1]))))
                y_max = int(min(out_h - 1, np.ceil(np.max(tq[:, 1]))))

                if x_max < x_min or y_max < y_min:
                    continue

                xs = np.arange(x_min, x_max + 1)
                ys = np.arange(y_min, y_max + 1)
                xv, yv = np.meshgrid(xs, ys)

                inside = point_in_triangle(
                    xv.astype(np.float32), yv.astype(np.float32), tq
                )
                if not np.any(inside):
                    continue

                x_flat = xv[inside].astype(np.float32)
                y_flat = yv[inside].astype(np.float32)

                # 应用仿射变换
                u = A[0, 0] * x_flat + A[0, 1] * y_flat + A[0, 2]
                v = A[1, 0] * x_flat + A[1, 1] * y_flat + A[1, 2]

                map_x[y_flat.astype(np.int32), x_flat.astype(np.int32)] = u
                map_y[y_flat.astype(np.int32), x_flat.astype(np.int32)] = v

                cell_count += 1

    return map_x, map_y


map_x, map_y = dense_map_from_grids(raw_ext, can_pts, out_h, OUT_WIDTH)

# 检查覆盖率
covered = (map_x >= 0).sum()
total = map_x.size
print(f"  处理了 {(N_ROWS+1)*(N_COLS+1)*2} 个三角形")
print(f"  覆盖率: {covered}/{total} = {100*covered/total:.1f}%")

if np.any(map_x < 0):
    uncovered = (map_x < 0).sum()
    print(f"  ⚠ 有 {uncovered} 个像素未覆盖，可增大 pad_px 解决")

# ============================================================
# Step 6: 保存 warp 并应用矫正
# ============================================================
print("\n【Step 6: 保存 warp 并应用矫正】")

# 保存 warp 为 npz
warp_path = os.path.join(OUTPUT_DIR, "demo_05_warp.npz")
meta = {
    "grid": {"n_cols": N_COLS, "n_rows": N_ROWS,
             "pitch_x_mm": PITCH_X_MM, "pitch_y_mm": PITCH_Y_MM},
    "rectified_px": {
        "out_width": OUT_WIDTH, "out_height": out_h,
        "pad_px": PAD_PX,
        "dx_raw_px": dx_raw_px, "dy_raw_px": dy_raw_px,
        "scale_x": scale_x, "scale_y": scale_y,
        "mm_per_px_x": PITCH_X_MM / (dx_raw_px * scale_x),
        "mm_per_px_y": PITCH_Y_MM / (dy_raw_px * scale_y),
    }
}
np.savez_compressed(warp_path, map_x=map_x, map_y=map_y, meta_json=json.dumps(meta))
print(f"  Warp 已保存到: {warp_path}")

# 应用矫正
rectified = cv2.remap(
    img, map_x, map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0
)
rectified_rgb = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)

# 裁剪到标记区域 + 应用底部遮挡蒙版
u0 = PAD_PX
v0 = PAD_PX
u1 = PAD_PX + (N_COLS - 1) * dx_rect
v1 = PAD_PX + (N_ROWS - 1) * dy_rect

x0 = max(0, int(np.floor(min(u0, u1) - PAD_PX)))
x1 = min(OUT_WIDTH, int(np.ceil(max(u0, u1) + PAD_PX)))
y0 = max(0, int(np.floor(min(v0, v1) - PAD_PX)))
# 底部遮挡：裁剪掉底部被遮挡的区域
y1 = min(out_h, int(np.ceil(max(v0, v1) + PAD_PX)) - MASK_BOTTOM_PX)

rectified_crop = rectified[y0:y1, x0:x1]
rectified_crop_rgb = cv2.cvtColor(rectified_crop, cv2.COLOR_BGR2RGB)

cv2.imwrite(os.path.join(OUTPUT_DIR, "demo_05_rectified.jpg"), rectified_crop)

# ============================================================
# Step 7: 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 原始图像
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original Image", fontsize=11)
axes[0, 0].axis("off")

# 原始图像 + 扩展网格
axes[0, 1].imshow(img_rgb)
for r in range(raw_ext.shape[0]):
    for c in range(raw_ext.shape[1] - 1):
        p0 = raw_ext[r, c]
        p1 = raw_ext[r, c + 1]
        axes[0, 1].plot([p0[0], p1[0]], [p0[1], p1[1]], "c-", linewidth=0.5)
for c in range(raw_ext.shape[1]):
    for r in range(raw_ext.shape[0] - 1):
        p0 = raw_ext[r, c]
        p1 = raw_ext[r + 1, c]
        axes[0, 1].plot([p0[0], p1[0]], [p0[1], p1[1]], "orange", linewidth=0.5)
axes[0, 1].set_title("Extended Grid (with boundary)", fontsize=11)
axes[0, 1].axis("off")

# 标准网格
axes[0, 2].scatter(can_pts[:, :, 0].ravel(), can_pts[:, :, 1].ravel(), s=1, c="blue")
for r in range(can_pts.shape[0]):
    for c in range(can_pts.shape[1] - 1):
        p0 = can_pts[r, c]
        p1 = can_pts[r, c + 1]
        axes[0, 2].plot([p0[0], p1[0]], [p0[1], p1[1]], "c-", linewidth=0.5)
for c in range(can_pts.shape[1]):
    for r in range(can_pts.shape[0] - 1):
        p0 = can_pts[r, c]
        p1 = can_pts[r + 1, c]
        axes[0, 2].plot([p0[0], p1[0]], [p0[1], p1[1]], "orange", linewidth=0.5)
axes[0, 2].set_title("Canonical Grid (uniform)", fontsize=11)
axes[0, 2].set_xlim(0, OUT_WIDTH)
axes[0, 2].set_ylim(out_h, 0)
axes[0, 2].set_aspect("equal")

# 矫正后图像
axes[1, 0].imshow(rectified_crop_rgb)
axes[1, 0].set_title("Rectified Image (cropped)", fontsize=11)
axes[1, 0].axis("off")

# 矫正后图像 + 网格 overlay
axes[1, 1].imshow(rectified_crop_rgb)
# 在裁剪后图像上绘制矫正网格
for r in range(N_ROWS):
    y_crop = PAD_PX + r * dy_rect - y0
    axes[1, 1].axhline(y=y_crop, color="cyan", linewidth=0.3, alpha=0.5)
for c in range(N_COLS):
    x_crop = PAD_PX + c * dx_rect - x0
    axes[1, 1].axvline(x=x_crop, color="orange", linewidth=0.3, alpha=0.5)
axes[1, 1].set_title("Rectified + Grid Overlay", fontsize=11)
axes[1, 1].axis("off")

# 矫正前后对比（并排）— 统一高度后拼接
common_h = min(rectified_crop_rgb.shape[0], int(OUT_WIDTH * img.shape[0] / img.shape[1]))
img_resized = cv2.resize(img_rgb, (OUT_WIDTH, common_h))
rect_resized = cv2.resize(rectified_crop_rgb, (OUT_WIDTH, common_h))
combined = np.hstack([img_resized, rect_resized])
axes[1, 2].imshow(combined)
axes[1, 2].set_title("Left: Original | Right: Rectified", fontsize=11)
axes[1, 2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "demo_05_warp.png"), dpi=150)
plt.close()
print(f"[保存] 可视化结果已保存到 {OUTPUT_DIR}/demo_05_warp.png")

# ============================================================
# Step 8: 总结
# ============================================================
print("\n" + "=" * 60)
print("【关键理解】")
print("  1. Warp = 稠密的 map_x/map_y 查找表，每个像素存原始图像对应坐标")
print("  2. 逐三角形仿射映射比全局 Homography 更精细，能处理非线性畸变")
print("  3. 标准网格保持物理纵横比，确保矫正后尺度一致")
print("  4. 边界外推确保矫正图像包含完整视场")
print("  5. 生成的 npz 文件可被后续所有处理步骤复用")
print(f"  6. mm_per_px ≈ {PITCH_X_MM / (dx_raw_px * scale_x):.4f} mm/px")
print("\n✅ demo_05 完成！")
print("  下一步: demo_06_compare.py — 对比分析")