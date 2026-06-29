"""
demo_08_warp_inpaint.py
Phase 2: 标记点跟踪 — 第2步
Warp 构建/加载 + 图像几何矫正 + Inpaint 标记点修复

学习目标：
  1. 理解：为什么要做标记点 Inpaint 修复？
     → 黑色标记点会干扰后续 Phase 3 的光度立体法向估计（颜色是法向的关键信息）
     → 需要在法向估计前把标记点"去掉"，用周围像素颜色插值填充
  2. 理解：Warp 矫正的作用
     → 将原始图像中扭曲的网格映射为横平竖直的标准网格
     → 矫正后标记点位置固定、间距均匀，便于后续处理
  3. 掌握：cv2.inpaint 的使用（Telea 算法）
  4. 对比：原始图像 → 矫正图像 → Inpaint修复后图像

算法流程（与 viewtacv2 一致）：
  raw image → detect markers → build warp → rectify → crop → inpaint markers
  （也可以在 raw 上先 inpaint 再 rectify，两种方式都演示）

注意：所有核心算法调用 tac_utils，参数与 Phase 1 完全一致。
"""

import cv2
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tac_utils
from tac_utils import N_COLS, N_ROWS

# ── 配置 ──────────────────────────────────────────────────
DATA_DIR = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
REF_FILE = "ref.jpg"
CONTACT_FILE = "image_20260430_135445_239.jpg"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
WARP_PATH = os.path.join(OUTPUT_DIR, "warp_ref.npz")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def dbg_save(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [保存] {path}")


# ── Step 1: 加载参考图像，构建/加载 Warp ────────────────────
print("=" * 60)
print("Step 1: 构建 Warp（基于参考图像 ref.jpg）")
print("=" * 60)

ref_path = os.path.join(DATA_DIR, REF_FILE)
img_ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)

if os.path.exists(WARP_PATH):
    print(f"  加载已有 Warp: {WARP_PATH}")
    map_x, map_y, meta = tac_utils.load_warp(WARP_PATH)
else:
    print("  从 ref.jpg 构建 Warp...")
    grid_ref, dog_ref, kp_ref = tac_utils.detect_and_sort(img_ref)
    n_valid = np.sum(~np.isnan(grid_ref[:, :, 0]))
    print(f"  检测到 {len(kp_ref)} 个点，有效网格点 {n_valid}/{N_COLS*N_ROWS}")

    map_x, map_y, meta, out_h = tac_utils.build_warp(grid_ref)
    tac_utils.save_warp(WARP_PATH, map_x, map_y, meta)
    print(f"  Warp 已保存: {WARP_PATH}")

rp = meta["rectified_px"]
out_h = rp["out_height"]
print(f"  矫正图像尺寸: {rp['out_width']}×{rp['out_height']}")
print(f"  矫正后间距: dx={rp['dx_rect']:.2f}px, dy={rp['dy_rect']:.2f}px")
print(f"  物理尺度: {rp['mm_per_px_x']:.5f} mm/px")


# ── Step 2: 矫正参考图像和接触图像 ─────────────────────────
print("\n" + "=" * 60)
print("Step 2: 应用 Warp 矫正图像")
print("=" * 60)

contact_path = os.path.join(DATA_DIR, CONTACT_FILE)
img_contact = cv2.imread(contact_path, cv2.IMREAD_COLOR)

rect_ref = tac_utils.apply_warp(img_ref, map_x, map_y)
rect_contact = tac_utils.apply_warp(img_contact, map_x, map_y)

rect_ref_crop, (x0, y0) = tac_utils.crop_rectified(rect_ref, meta)
rect_contact_crop, _ = tac_utils.crop_rectified(rect_contact, meta)

print(f"  参考图像矫正裁剪后尺寸: {rect_ref_crop.shape}")
print(f"  接触图像矫正裁剪后尺寸: {rect_contact_crop.shape}")


# ── Step 3: Inpaint 标记点修复 ─────────────────────────────
print("\n" + "=" * 60)
print("Step 3: Inpaint 标记点修复")
print("=" * 60)

# 方式1：在矫正后图像上修复（标准网格位置，遮罩更精确）
rect_ref_inp = tac_utils.inpaint_rectified(rect_ref, meta)
rect_contact_inp = tac_utils.inpaint_rectified(rect_contact, meta)

rect_ref_inp_crop, _ = tac_utils.crop_rectified(rect_ref_inp, meta)
rect_contact_inp_crop, _ = tac_utils.crop_rectified(rect_contact_inp, meta)

# 方式2：在原始图像上先检测标记点再修复，然后矫正
grid_ref, _, kp_ref = tac_utils.detect_and_sort(img_ref)
grid_contact, _, kp_contact = tac_utils.detect_and_sort(img_contact)

img_ref_inp_raw = tac_utils.inpaint_raw(img_ref, grid_ref, marker_radius=5.0)
img_contact_inp_raw = tac_utils.inpaint_raw(img_contact, grid_contact, marker_radius=5.0)

rect_ref_inp2 = tac_utils.apply_warp(img_ref_inp_raw, map_x, map_y)
rect_contact_inp2 = tac_utils.apply_warp(img_contact_inp_raw, map_x, map_y)
rect_ref_inp2_crop, _ = tac_utils.crop_rectified(rect_ref_inp2, meta)
rect_contact_inp2_crop, _ = tac_utils.crop_rectified(rect_contact_inp2, meta)

print("  方式1: 先矫正 → 再 inpaint（使用标准网格位置）")
print("  方式2: 先 inpaint → 再矫正（使用检测到的标记点位置）")


# ── Step 4: 可视化对比 ─────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: 可视化对比")
print("=" * 60)

img_rgb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
img_rgb_contact = cv2.cvtColor(img_contact, cv2.COLOR_BGR2RGB)
rect_ref_rgb = cv2.cvtColor(rect_ref_crop, cv2.COLOR_BGR2RGB)
rect_contact_rgb = cv2.cvtColor(rect_contact_crop, cv2.COLOR_BGR2RGB)
rect_ref_inp_rgb = cv2.cvtColor(rect_ref_inp_crop, cv2.COLOR_BGR2RGB)
rect_contact_inp_rgb = cv2.cvtColor(rect_contact_inp_crop, cv2.COLOR_BGR2RGB)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第1行：参考图像
axes[0, 0].imshow(img_rgb_ref)
axes[0, 0].set_title("原始参考图像 (Raw Ref)", fontsize=11)
axes[0, 0].axis("off")

axes[0, 1].imshow(rect_ref_rgb)
axes[0, 1].set_title("Warp 矫正后 (Rectified)", fontsize=11)
axes[0, 1].axis("off")

axes[0, 2].imshow(rect_ref_inp_rgb)
axes[0, 2].set_title("Inpaint 修复后 (Markers Removed)", fontsize=11)
axes[0, 2].axis("off")

# 第2行：接触图像
axes[1, 0].imshow(img_rgb_contact)
axes[1, 0].set_title(f"原始接触图像\n({CONTACT_FILE})", fontsize=11)
axes[1, 0].axis("off")

axes[1, 1].imshow(rect_contact_rgb)
axes[1, 1].set_title("Warp 矫正后 (Rectified)", fontsize=11)
axes[1, 1].axis("off")

axes[1, 2].imshow(rect_contact_inp_rgb)
axes[1, 2].set_title("Inpaint 修复后 (Markers Removed)", fontsize=11)
axes[1, 2].axis("off")

plt.suptitle("Phase 2 Step 2: Warp 矫正 + Inpaint 标记修复", fontsize=14, fontweight='bold')
plt.tight_layout()
dbg_save(fig, "demo_08_warp_inpaint")

# ── Step 5: Inpaint 效果放大对比 ───────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 12))

# 选取中间区域放大
h, w = rect_ref_rgb.shape[:2]
crop_y0, crop_y1 = int(h*0.3), int(h*0.7)
crop_x0, crop_x1 = int(w*0.2), int(w*0.8)

zoom_ref = rect_ref_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
zoom_ref_inp = rect_ref_inp_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
zoom_contact = rect_contact_rgb[crop_y0:crop_y1, crop_x0:crop_x1]
zoom_contact_inp = rect_contact_inp_rgb[crop_y0:crop_y1, crop_x0:crop_x1]

axes2[0, 0].imshow(zoom_ref)
axes2[0, 0].set_title("Ref 矫正后（放大，可见黑色标记点）", fontsize=11)
axes2[0, 0].axis("off")

axes2[0, 1].imshow(zoom_ref_inp)
axes2[0, 1].set_title("Ref Inpaint后（标记点已去除）", fontsize=11)
axes2[0, 1].axis("off")

axes2[1, 0].imshow(zoom_contact)
axes2[1, 0].set_title("Contact 矫正后（放大，可见黑色标记点）", fontsize=11)
axes2[1, 0].axis("off")

axes2[1, 1].imshow(zoom_contact_inp)
axes2[1, 1].set_title("Contact Inpaint后（标记点已去除）", fontsize=11)
axes2[1, 1].axis("off")

plt.suptitle("Inpaint 效果放大对比（中间区域）", fontsize=14, fontweight='bold')
plt.tight_layout()
dbg_save(fig2, "demo_08_inpaint_zoom")

# ── Step 6: 总结 ──────────────────────────────────────────
print("\n" + "=" * 60)
print("【关键理解】")
print("  1. Warp 矫正：将原始扭曲图像映射为横平竖直的标准网格图像")
print("  2. Inpaint 修复：用周围像素插值填充黑色标记点区域")
print("     → 为什么要修复？标记点是黑色的，会严重干扰光度立体的颜色分析")
print("     → 光度立体依赖 RGB 三色光照下的颜色比例来推算法向量")
print("     → 黑色标记点没有颜色信息，必须去掉才能做后续处理")
print("  3. 两种 inpaint 方式：")
print("     → 方式1（推荐）：先矫正再 inpaint，标准网格位置固定，遮罩更准确")
print("     → 方式2：先 inpaint 再矫正，适用于需要在原始图像上处理的场景")
print(f"  4. 物理尺度: {rp['mm_per_px_x']:.5f} mm/px（矫正后图像）")
print("\n✅ demo_08 完成！")
print("  下一步: demo_09_displacement_heatmap.py — 矫正后位移场计算与热力图")
