"""
demo_03_pca_sort.py — PCA 网格排序
===================================
目的：理解如何将无序的检测点排序为规则的 (row, col) 网格。

算法原理：
  DoG 检测到的标记点是无序的。但物理上它们是规则的网格。
  我们需要把每个点映射到它的 (row, col) 索引。

  核心思路：
  1. PCA 找到点集的两个主方向（≈ 行方向和列方向）
  2. 将点投影到这两个主轴上
  3. 用 k-means 聚类分别对行/列投影坐标分组
  4. 贪心分配：每个点分配到最近的 (row, col) 格子

  为什么需要 PCA？
  - 相机与凝胶表面不平行，网格在图像中可能是倾斜的
  - PCA 自动找到点集的主方向，不需要手动指定旋转角度

  为什么需要 k-means？
  - 透视效应导致行间距不均匀（近大远小）
  - k-means 能自适应地找到每行/列的聚类中心

运行：python demo_03_pca_sort.py
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

# 网格规格
# 注意：当前图像（data_wenli/ref.jpg）相比旧 calibData 图像旋转了 90°
# 旧图像：网格 16列×21行，横向展开，左侧有遮挡（mask_left_px=1300）
# 新图像：网格 16列×21行，竖向展开，底部有遮挡（mask_bottom_px）
# 实测：ref.jpg 检测到约 320 个点（16×20），底部一行可能被遮挡
N_COLS = 16
N_ROWS = 21  # 物理网格是 21 行
N_EXPECTED = N_COLS * N_ROWS  # 336

# ============================================================
# ROI（有效区域）遮罩
# ----------------------------------------------------------
# 为什么需要 ROI？
#   原始图像中除了传感器凝胶区域，还有：顶部 LED 灯、黑色边框、底部反光金属。
#   这些区域在 DoG 后也会产生亮斑，被 blob 检测器误检为标记点，干扰 k-means 聚类。
#   所以在检测前，把 ROI 外的像素设为 0（黑色），让检测器只关注有效区域。
# 如何确定 ROI？
#   通过观察图像中标记点的分布范围，预留少量边距得到。
# ============================================================
ROI_X0, ROI_Y0 = 810, 225
ROI_X1, ROI_Y1 = 1140, 630

# ============================================================
# 1. 读取图像并检测标记点
# ============================================================
img_path = os.path.join(DATA_DIR, IMAGE_FILE)
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# DoG 预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
blur_large = cv2.GaussianBlur(gray, (27, 27), 0.3 * ((27 - 1) * 0.5 - 1) + 0.8)
blur_small = cv2.GaussianBlur(gray, (5, 5), 0.3 * ((5 - 1) * 0.5 - 1) + 0.8)
dog = np.clip((blur_large - blur_small) * 12.9, 0.0, 255.0).astype(np.uint8)

# 应用 ROI 遮罩：ROI 外的像素设为 0，防止误检
roi_mask = np.zeros_like(dog)
roi_mask[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1] = 255
dog = cv2.bitwise_and(dog, roi_mask)

# Blob 检测
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 171.0  # 降低阈值，确保低对比度下稳定检测到 336 个点
params.maxThreshold = 255.0
params.thresholdStep = 11.0
params.minDistBetweenBlobs = 6.0
params.filterByArea = True
params.minArea = 8.0
params.maxArea = 1500.0
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByConvexity = False  # 关闭凸性过滤，与 demo_02 保持一致
params.filterByInertia = False    # 关闭惯性过滤，与 demo_02 保持一致
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(dog)
keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)

# 取最强的前 N_EXPECTED 个点
keypoints = keypoints[:N_EXPECTED]
points = np.array([(float(k.pt[0]), float(k.pt[1])) for k in keypoints], dtype=np.float64)

print("=" * 60)
print("【PCA 网格排序 — 逐步演示】")
print(f"  检测到 {len(keypoints)} 个关键点（取前 {N_EXPECTED}）")

# ============================================================
# 2. PCA：找到点集的主方向
# ============================================================
print("\n【Step 1: PCA 主方向分析】")

# 计算均值并中心化
mean = points.mean(axis=0)
centered = points - mean

# 计算协方差矩阵(点集在二维平面上的"椭圆形状"") 
cov = np.cov(centered.T)
print(f"  协方差矩阵:\n{cov}")

# 特征分解
eigvals, eigvecs = np.linalg.eigh(cov)
print(f"  特征值: {eigvals}")
print(f"  特征向量:\n{eigvecs}")

# 按特征值降序排列
order = np.argsort(eigvals)[::-1]
R = eigvecs[:, order]  # 旋转矩阵：第一列 = 最大方差方向

# 投影到主成分空间
proj = centered @ R  # (N, 2)

# 确保主轴方向与图像坐标系一致（x向右，y向下）
if np.corrcoef(proj[:, 0], centered[:, 0])[0, 1] < 0:
    proj[:, 0] *= -1
    print("  [校正] 翻转 PC1 使其与图像 x 方向一致")
if np.corrcoef(proj[:, 1], centered[:, 1])[0, 1] < 0:
    proj[:, 1] *= -1
    print("  [校正] 翻转 PC2 使其与图像 y 方向一致")

# 关键：判断哪个主成分对应列方向（水平），哪个对应行方向（垂直）
# 原始图像（calibData）：PC1≈水平(x)，PC2≈垂直(y)，与期望一致
# 新图像（data_wenli，旋转90°）：PC1≈垂直(y)，PC2≈水平(x)，需要交换！
# 判断方法：看哪个主成分的 x 分量绝对值更大 → 更水平
pc1_is_horizontal = abs(R[0, 0]) > abs(R[0, 1])
if pc1_is_horizontal:
    col_base = proj[:, 0]  # PC1 ≈ 列方向（水平）
    row_base = proj[:, 1]  # PC2 ≈ 行方向（垂直）
    print("  [检测] PC1 为水平方向 → 列方向")
else:
    col_base = proj[:, 1]  # PC2 ≈ 列方向（水平）
    row_base = proj[:, 0]  # PC1 ≈ 行方向（垂直）
    print("  [检测] PC1 为垂直方向 → 行方向（图像旋转了90°）")

print(f"  列方向 范围: [{col_base.min():.0f}, {col_base.max():.0f}]")
print(f"  行方向 范围: [{row_base.min():.0f}, {row_base.max():.0f}]")

# ============================================================
# 3. k-means 聚类：分别对行/列投影坐标分组
# ============================================================
print("\n【Step 2: k-means 聚类分组】")

def kmeans_1d(values, k, iters=50):
    """一维 k-means 聚类（初始中心在 min~max 间均匀分布，适合等间距网格）"""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    # 注意：不能用 quantile(0.05~0.95) 初始化——当边缘点少时，两端聚类中心会挤在一起
    # 用 linspace(min, max, k) 均匀初始化，正好利用已知的"等间距网格"先验
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

print(f"  列聚类中心 ({N_COLS} 个): {np.round(x_centers, 0).tolist()}")
print(f"  行聚类中心 ({N_ROWS} 个): {np.round(y_centers, 0).tolist()}")
print(f"  列间距: {np.round(np.diff(x_centers), 1).tolist()}")
print(f"  行间距: {np.round(np.diff(y_centers), 1).tolist()}")

# ============================================================
# 4. 贪心分配：每个点分配到最近的 (row, col)
# ============================================================
print("\n【Step 3: 贪心分配】")

# 每个点找到最近的聚类中心 Numpy 广播计算距离 广播是一个非常强大的工具，它可以让我们在不使用显式循环的情况下对数组进行操作。
# 这里我们使用广播来计算每个点到所有聚类中心的距离，从而找到最近的聚类中心。
col_idx = np.argmin(np.abs(col_base[:, None] - x_centers[None, :]), axis=1)
row_idx = np.argmin(np.abs(row_base[:, None] - y_centers[None, :]), axis=1)

# 方向校正：确保 col 从左到右递增，row 从上到下递增
# PCA 的特征向量方向是任意的（可以正负翻转），所以需要用图像坐标来校正
if np.corrcoef(col_idx, points[:, 0])[0, 1] < 0:
    col_idx = N_COLS - 1 - col_idx
    x_centers = x_centers[::-1]
if np.corrcoef(row_idx, points[:, 1])[0, 1] < 0:
    row_idx = N_ROWS - 1 - row_idx
    y_centers = y_centers[::-1]

# 计算分配代价（距离聚类中心的远近）
costs = (col_base - x_centers[col_idx]) ** 2 + (row_base - y_centers[row_idx]) ** 2

# 按代价排序，优先分配代价低的点
order_pts = np.argsort(costs)

# 初始化空网格
grid_points = np.full((N_ROWS, N_COLS, 2), np.nan, dtype=np.float64)

for i in order_pts:
    r = int(row_idx[i])
    c = int(col_idx[i])
    if 0 <= r < N_ROWS and 0 <= c < N_COLS and np.isnan(grid_points[r, c, 0]):
        grid_points[r, c] = points[i]

# 统计每行分配的点数
row_counts = np.sum(~np.isnan(grid_points[:, :, 0]), axis=1)
print(f"  每行分配点数: {row_counts.tolist()}")

empty_cells = np.argwhere(np.isnan(grid_points[:, :, 0]))
print(f"  空单元格数: {len(empty_cells)}")

if len(empty_cells) > 0:
    print(f"  空单元格位置: {empty_cells.tolist()}")
    print("  ⚠ 存在空单元格，可能需要调整检测参数或使用 rank-based 兜底")

# ============================================================
# 5. 验证网格拓扑
# ============================================================
print("\n【Step 4: 拓扑验证】")

issues = []

# 检查每行 x 是否单调递增
for r in range(N_ROWS):
    row = grid_points[r, :, 0]
    if not np.isnan(row).any():
        dx = np.diff(row)
        if np.mean(dx < 0) > 0.35:
            issues.append(f"row {r}: x decreases")

# 检查每列 y 是否单调递增
for c in range(N_COLS):
    col = grid_points[:, c, 1]
    if not np.isnan(col).any():
        dy = np.diff(col)
        if np.mean(dy < 0) > 0.35:
            issues.append(f"col {c}: y decreases")

if issues:
    print(f"  ⚠ 拓扑问题: {issues}")
else:
    print("  ✅ 网格拓扑正常")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 原始图像 + 检测点
axes[0, 0].imshow(img_rgb)
axes[0, 0].scatter(points[:, 0], points[:, 1], c="red", s=2, alpha=0.5)
axes[0, 0].set_title(f"Detected Points ({len(points)})", fontsize=11)
axes[0, 0].axis("off")

# PCA 投影空间
axes[0, 1].scatter(col_base, row_base, c="blue", s=3, alpha=0.6)
for xc in x_centers:
    axes[0, 1].axvline(x=xc, color="red", linestyle="--", alpha=0.5)
for yc in y_centers:
    axes[0, 1].axhline(y=yc, color="green", linestyle="--", alpha=0.5)
axes[0, 1].set_title("PCA Projection + k-means Centers", fontsize=11)
axes[0, 1].set_xlabel("PC1 (column direction)")
axes[0, 1].set_ylabel("PC2 (row direction)")

# 排序后的网格（原始图像坐标）
axes[0, 2].imshow(img_rgb)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_points[r, c, 0]):
            u, v = grid_points[r, c]
            axes[0, 2].plot(u, v, "o", markersize=3, color="lime")
            if r == 0 or r == N_ROWS - 1 or c == 0 or c == N_COLS - 1:
                idx = r * N_COLS + c
                axes[0, 2].text(u + 3, v - 3, str(idx), fontsize=5, color="yellow")
axes[0, 2].set_title("Indexed Grid (raw image coords)", fontsize=11)
axes[0, 2].axis("off")

# 网格线（显示行/列连接）
axes[1, 0].imshow(img_rgb)
for r in range(N_ROWS):
    for c in range(N_COLS - 1):
        if not np.isnan(grid_points[r, c, 0]) and not np.isnan(grid_points[r, c + 1, 0]):
            p0 = grid_points[r, c]
            p1 = grid_points[r, c + 1]
            axes[1, 0].plot([p0[0], p1[0]], [p0[1], p1[1]], "c-", linewidth=0.5, alpha=0.8)
for c in range(N_COLS):
    for r in range(N_ROWS - 1):
        if not np.isnan(grid_points[r, c, 0]) and not np.isnan(grid_points[r + 1, c, 0]):
            p0 = grid_points[r, c]
            p1 = grid_points[r + 1, c]
            axes[1, 0].plot([p0[0], p1[0]], [p0[1], p1[1]], "orange", linewidth=0.5, alpha=0.8)
axes[1, 0].set_title("Grid Connections (cyan=row, orange=col)", fontsize=11)
axes[1, 0].axis("off")

# 行间距分布
row_spacings = []
for r in range(N_ROWS - 1):
    for c in range(N_COLS):
        if not np.isnan(grid_points[r, c, 0]) and not np.isnan(grid_points[r + 1, c, 0]):
            d = np.linalg.norm(grid_points[r + 1, c] - grid_points[r, c])
            row_spacings.append(d)

axes[1, 1].plot(row_spacings, "b.-", alpha=0.7)
axes[1, 1].axhline(y=np.median(row_spacings), color="r", linestyle="--", label=f"median={np.median(row_spacings):.1f}")
axes[1, 1].set_title("Row Spacings (pixel distance)", fontsize=11)
axes[1, 1].set_xlabel("Row pair index")
axes[1, 1].set_ylabel("Distance (px)")
axes[1, 1].legend()

# 列间距分布
col_spacings = []
for c in range(N_COLS - 1):
    for r in range(N_ROWS):
        if not np.isnan(grid_points[r, c, 0]) and not np.isnan(grid_points[r, c + 1, 0]):
            d = np.linalg.norm(grid_points[r, c + 1] - grid_points[r, c])
            col_spacings.append(d)

axes[1, 2].plot(col_spacings, "g.-", alpha=0.7)
axes[1, 2].axhline(y=np.median(col_spacings), color="r", linestyle="--", label=f"median={np.median(col_spacings):.1f}")
axes[1, 2].set_title("Column Spacings (pixel distance)", fontsize=11)
axes[1, 2].set_xlabel("Column pair index")
axes[1, 2].set_ylabel("Distance (px)")
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "demo_03_pca_sort_new3.png"), dpi=150)
plt.close()
print(f"\n[保存] 可视化结果已保存到 {OUTPUT_DIR}/demo_03_pca_sort.png")

# ============================================================
# 7. 总结
# ============================================================
print("\n" + "=" * 60)
print("【关键理解】")
print("  1. PCA 自动找到点集的主方向，不需要手动指定旋转角度")
print("  2. k-means 聚类将投影坐标分为 N_ROWS 组和 N_COLS 组")
print("  3. 贪心分配确保每个 (row, col) 位置只有一个点")
print("  4. 行/列间距不均匀是正常的（透视效应），这正是需要矫正的原因")
print("  5. 如果空单元格太多，说明检测参数需要调整")
print("\n✅ demo_03 完成！")
print("  下一步: demo_04_homography.py — 学习透视矫正")