"""
demo_07_two_image_detect.py
Phase 2: 标记点跟踪 — 第1步
同时对参考图像和接触图像进行标记点检测与网格排序，对比两图差异。

学习目标：
  1. 理解：按压前后，标记点位置发生了变化
  2. 观察：哪些区域变化大（接触区），哪些区域变化小（远离接触区）
  3. 为下一步位移计算（demo_08）做准备

数据来源：D:\\02_Life-Long Learning\\Project\\01_Visual_tactile\\tactile\\data\\data_wenli
  - ref.jpg: 参考图像（无接触）
  - image_*.jpg: 400张接触图像

注意：检测+排序逻辑完全复用 tac_utils（与 Phase 1 一致）
"""

import cv2
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os
import sys

# ── 导入公共模块（与 Phase 1 完全一致） ───────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tac_utils
from tac_utils import N_COLS, N_ROWS, N_EXPECTED

# ── 配置 ──────────────────────────────────────────────────
DATA_DIR  = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
REF_FILE  = "ref.jpg"

# 接触图像：选第1张试试效果
CONTACT_FILE = "image_20260430_135445_239.jpg"

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 调试保存辅助
def dbg_save(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [保存] {path}")


# ── Step 1: 加载图像 ──────────────────────────────────────
print("=" * 60)
print("Step 1: 加载图像")
print("=" * 60)

ref_path    = os.path.join(DATA_DIR, REF_FILE)
contact_path = os.path.join(DATA_DIR, CONTACT_FILE)

img_ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
img_contact = cv2.imread(contact_path, cv2.IMREAD_COLOR)

if img_ref is None:
    print(f"错误: 找不到参考图像 {ref_path}")
    sys.exit(1)
if img_contact is None:
    print(f"错误: 找不到接触图像 {contact_path}")
    sys.exit(1)

print(f"参考图像: {REF_FILE}  {img_ref.shape}")
print(f"接触图像: {CONTACT_FILE}  {img_contact.shape}")


# ── Step 2: 检测 + 排序（调用 tac_utils，与 Phase 1 完全一致）──
print("\n" + "=" * 60)
print("Step 2: 检测 + 网格排序（调用 tac_utils）")
print("=" * 60)

grid_ref, dog_ref, kp_ref = tac_utils.detect_and_sort(img_ref)
grid_contact, dog_contact, kp_contact = tac_utils.detect_and_sort(img_contact)

img_rgb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
img_rgb_contact = cv2.cvtColor(img_contact, cv2.COLOR_BGR2RGB)

n_empty_ref = np.sum(np.isnan(grid_ref[:, :, 0]))
n_empty_contact = np.sum(np.isnan(grid_contact[:, :, 0]))
print(f"\n参考图像: 检测到 {len(kp_ref)} 个点, 空单元格 {n_empty_ref}")
print(f"接触图像: 检测到 {len(kp_contact)} 个点, 空单元格 {n_empty_contact}")


# ── Step 3: 双图对比可视化 ─────────────────────────────────
print("\n" + "=" * 60)
print("Step 3: 双图对比可视化")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 第1行：参考图像
axes[0, 0].imshow(img_rgb_ref)
axes[0, 0].set_title("Reference Image (ref.jpg)", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(img_rgb_ref)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_ref[r, c, 0]):
            axes[0, 1].plot(grid_ref[r, c, 0], grid_ref[r, c, 1], 'o', ms=3, color='lime')
axes[0, 1].set_title(f"Reference Grid {N_ROWS}x{N_COLS} (empty={n_empty_ref})", fontsize=12)
axes[0, 1].axis("off")

axes[0, 2].imshow(img_rgb_ref)
for r in range(N_ROWS):
    xs = [grid_ref[r, c, 0] for c in range(N_COLS) if not np.isnan(grid_ref[r, c, 0])]
    ys = [grid_ref[r, c, 1] for c in range(N_COLS) if not np.isnan(grid_ref[r, c, 0])]
    if len(xs) > 1:
        axes[0, 2].plot(xs, ys, 'lime', linewidth=0.5, alpha=0.6)
for c in range(N_COLS):
    xs = [grid_ref[r, c, 0] for r in range(N_ROWS) if not np.isnan(grid_ref[r, c, 0])]
    ys = [grid_ref[r, c, 1] for r in range(N_ROWS) if not np.isnan(grid_ref[r, c, 0])]
    if len(xs) > 1:
        axes[0, 2].plot(xs, ys, 'lime', linewidth=0.5, alpha=0.6)
axes[0, 2].set_title("Reference Grid Lines", fontsize=12)
axes[0, 2].axis("off")

# 第2行：接触图像
axes[1, 0].imshow(img_rgb_contact)
axes[1, 0].set_title(f"Contact Image ({CONTACT_FILE})", fontsize=12)
axes[1, 0].axis("off")

axes[1, 1].imshow(img_rgb_contact)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_contact[r, c, 0]):
            axes[1, 1].plot(grid_contact[r, c, 0], grid_contact[r, c, 1], 'o', ms=3, color='lime')
axes[1, 1].set_title(f"Contact Grid {N_ROWS}x{N_COLS} (empty={n_empty_contact})", fontsize=12)
axes[1, 1].axis("off")

axes[1, 2].imshow(img_rgb_contact)
for r in range(N_ROWS):
    xs = [grid_contact[r, c, 0] for c in range(N_COLS) if not np.isnan(grid_contact[r, c, 0])]
    ys = [grid_contact[r, c, 1] for c in range(N_COLS) if not np.isnan(grid_contact[r, c, 0])]
    if len(xs) > 1:
        axes[1, 2].plot(xs, ys, 'lime', linewidth=0.5, alpha=0.6)
for c in range(N_COLS):
    xs = [grid_contact[r, c, 0] for r in range(N_ROWS) if not np.isnan(grid_contact[r, c, 0])]
    ys = [grid_contact[r, c, 1] for r in range(N_ROWS) if not np.isnan(grid_contact[r, c, 0])]
    if len(xs) > 1:
        axes[1, 2].plot(xs, ys, 'lime', linewidth=0.5, alpha=0.6)
axes[1, 2].set_title("Contact Grid Lines", fontsize=12)
axes[1, 2].axis("off")

plt.tight_layout()
dbg_save(fig, "demo_07_two_image_grids")

# ── Step 4: 叠加对比 — 参考 vs 接触 ───────────────────────
print("\n" + "=" * 60)
print("Step 4: 叠加对比 (参考 vs 接触)")
print("=" * 60)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 7))

# 左图：在接触图像上同时画参考网格（红）和接触网格（绿）
axes2[0].imshow(img_rgb_contact)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_ref[r, c, 0]):
            axes2[0].plot(grid_ref[r, c, 0], grid_ref[r, c, 1], 'o', ms=3, color='red', alpha=0.5)
        if not np.isnan(grid_contact[r, c, 0]):
            axes2[0].plot(grid_contact[r, c, 0], grid_contact[r, c, 1], 'o', ms=3, color='lime', alpha=0.5)
axes2[0].set_title("Contact Image + Red=Ref / Green=Contact", fontsize=12)
axes2[0].axis("off")

# 右图：位移向量（放大2倍便于观察）
axes2[1].imshow(img_rgb_contact)
n_valid = 0
displacements = []
valid_pairs = []
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_ref[r, c, 0]) and not np.isnan(grid_contact[r, c, 0]):
            dx = grid_contact[r, c, 0] - grid_ref[r, c, 0]
            dy = grid_contact[r, c, 1] - grid_ref[r, c, 1]
            axes2[1].arrow(grid_ref[r, c, 0], grid_ref[r, c, 1],
                           dx * 2, dy * 2,
                           head_width=2, head_length=3, fc='yellow', ec='yellow',
                           alpha=0.7, linewidth=0.5)
            displacements.append(np.linalg.norm([dx, dy]))
            valid_pairs.append((r, c))
            n_valid += 1

displacements = np.array(displacements)
axes2[1].set_title(f"Displacement Vectors (2x)  n={n_valid}  "
                   f"mean={displacements.mean():.2f}px  max={displacements.max():.2f}px",
                   fontsize=12)
axes2[1].axis("off")

plt.tight_layout()
dbg_save(fig2, "demo_07_overlay_displacement")

# ── Step 5: 统计输出 ──────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5: 统计输出")
print("=" * 60)

print(f"\n参考图像检测点数: {len(kp_ref)}")
print(f"接触图像检测点数: {len(kp_contact)}")
print(f"有效配对点数: {n_valid} / {N_EXPECTED}")

if len(displacements) > 0:
    print(f"\n位移统计:")
    print(f"  均值: {displacements.mean():.2f} px")
    print(f"  标准差: {displacements.std():.2f} px")
    print(f"  最大值: {displacements.max():.2f} px")
    print(f"  最小值: {displacements.min():.2f} px")
    print(f"  中位数: {np.median(displacements):.2f} px")

    # 位移最大的5个点
    top_idx = np.argsort(displacements)[::-1][:5]
    print(f"\n位移最大的5个点:")
    for idx in top_idx:
        r, c = valid_pairs[idx]
        print(f"  [{r},{c}] 位移: {displacements[idx]:.2f} px")

print(f"\n输出图片保存在: {OUTPUT_DIR}")
print("=" * 60)
print("Phase 2 Step 1 完成！")
print("下一步: demo_08 逐点位移计算与可视化")
print("\n  注：检测+排序逻辑完全复用 tac_utils，与 Phase 1 保持一致。")
