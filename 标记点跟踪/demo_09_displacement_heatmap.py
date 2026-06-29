"""
demo_09_displacement_heatmap.py — Phase 2 Step 3: 矫正后位移场计算与热力图可视化

本 demo 实现:
  1. 加载/构建 warp（含三角剖分信息）
  2. 在原始图像上检测 ref 和 contact 的网格点（使用 Phase 1 已验证的方法）
  3. 通过三角剖分逆变换将网格点正向映射到矫正后坐标
  4. 计算矫正后坐标下的位移场（像素→mm）
  5. 热力图可视化（dx, dy, |d|）+ 箭头图

为什么要用正向点映射？
  - 原始图像上的检测+排序已经在 Phase 1 验证过，稳定可靠
  - warp 的三角剖分给出 can→raw 仿射变换，逆变换就是 raw→can
  - 矫正后坐标下网格横平竖直，mm/px 均匀，位移物理意义明确
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
REF_PATH = os.path.join(DATA_DIR, "ref.jpg")
WARP_PATH = os.path.join(os.path.dirname(__file__), "outputs", "warp_ref.npz")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

CONTACT_FILENAME = "image_20260430_135445_239.jpg"
CONTACT_PATH = os.path.join(DATA_DIR, CONTACT_FILENAME)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 1: 构建/加载 Warp（含三角剖分）")
    print("=" * 60)

    img_ref = cv2.imread(REF_PATH)
    assert img_ref is not None, f"无法读取 {REF_PATH}"

    # 每次重新构建 warp（确保 tris 信息存在）
    print("  从 ref.jpg 重建 Warp（包含三角剖分逆变换）...")
    grid_ref_raw, dog_ref_raw, kp_ref_raw = tac_utils.detect_and_sort(img_ref)
    n_valid = np.sum(~np.isnan(grid_ref_raw[:, :, 0]))
    print(f"  检测到 {len(kp_ref_raw)} 个点，有效网格点 {n_valid}/{tac_utils.N_COLS*tac_utils.N_ROWS}")

    map_x, map_y, meta, out_h = tac_utils.build_warp(grid_ref_raw)
    tac_utils.save_warp(WARP_PATH, map_x, map_y, meta)
    print(f"  Warp 已保存: {WARP_PATH}")

    rp = meta["rectified_px"]
    mm_per_px = rp["mm_per_px_x"]
    print(f"  矫正尺寸: {rp['out_width']}×{rp['out_height']}")
    print(f"  矫正后间距: dx={rp['dx_rect']:.2f}px, dy={rp['dy_rect']:.2f}px")
    print(f"  物理尺度: {mm_per_px:.5f} mm/px (x), {rp['mm_per_px_y']:.5f} mm/px (y)")
    print(f"  三角形数量: {len(meta['tris'])}")

    # 标准网格（ref 在矫正后全图中的位置）
    grid_ref_rect_theory, dx_rect, dy_rect = tac_utils.get_rectified_grid_centers(meta)

    print("=" * 60)
    print("Step 2: 在原始图像上检测 contact 网格点")
    print("=" * 60)

    img_contact = cv2.imread(CONTACT_PATH)
    assert img_contact is not None, f"无法读取 {CONTACT_PATH}"

    grid_contact_raw, dog_contact_raw, kp_contact_raw = tac_utils.detect_and_sort(img_contact)
    n_contact = np.sum(~np.isnan(grid_contact_raw[:, :, 0]))
    print(f"  contact 检测到 {len(kp_contact_raw)} 个点，有效网格点 {n_contact}/{tac_utils.N_COLS*tac_utils.N_ROWS}")

    print("=" * 60)
    print("Step 3: 正向映射到矫正后坐标")
    print("=" * 60)

    # 将 ref 和 contact 的网格点都映射到矫正后坐标
    grid_ref_rect = tac_utils.warp_points_raw_to_rect(grid_ref_raw, meta)
    grid_contact_rect = tac_utils.warp_points_raw_to_rect(grid_contact_raw, meta)

    n_ref_rect = np.sum(~np.isnan(grid_ref_rect[:, :, 0]))
    n_contact_rect = np.sum(~np.isnan(grid_contact_rect[:, :, 0]))
    print(f"  ref 映射后有效点: {n_ref_rect}")
    print(f"  contact 映射后有效点: {n_contact_rect}")

    # 验证 ref 映射后位置与理论标准网格位置的偏差
    ref_err = grid_ref_rect - grid_ref_rect_theory
    ref_err_mag = np.sqrt(ref_err[:, :, 0]**2 + ref_err[:, :, 1]**2)
    valid = ~np.isnan(ref_err_mag)
    if np.sum(valid) > 0:
        print(f"  ref 映射位置与标准网格偏差: mean={np.nanmean(ref_err_mag[valid]):.3f}px, "
              f"max={np.nanmax(ref_err_mag[valid]):.3f}px")

    print("=" * 60)
    print("Step 4: 计算位移场（像素 → mm）")
    print("=" * 60)

    # 位移 = contact 矫正后位置 - ref 矫正后理论位置
    disp_px = grid_contact_rect - grid_ref_rect_theory
    valid_mask = ~(np.isnan(disp_px[:, :, 0]) | np.isnan(disp_px[:, :, 1]))
    disp_px[~valid_mask] = np.nan
    disp_mm = tac_utils.disp_to_physical(disp_px, meta)

    dx_mm = disp_mm[:, :, 0]
    dy_mm = disp_mm[:, :, 1]
    disp_mag = np.sqrt(dx_mm**2 + dy_mm**2)

    vmax = float(np.nanpercentile(disp_mag[valid_mask], 95)) if np.sum(valid_mask) > 0 else 0.1
    print(f"  有效点数: {np.sum(valid_mask)}/{tac_utils.N_COLS*tac_utils.N_ROWS}")
    print(f"  最大位移 |d| (95%分位): {vmax:.4f} mm")
    print(f"  dx 范围: [{np.nanmin(dx_mm[valid_mask]):.4f}, {np.nanmax(dx_mm[valid_mask]):.4f}] mm")
    print(f"  dy 范围: [{np.nanmin(dy_mm[valid_mask]):.4f}, {np.nanmax(dy_mm[valid_mask]):.4f}] mm")

    print("=" * 60)
    print("Step 5: Warp 矫正图像（用于可视化背景）")
    print("=" * 60)

    rect_contact_full = tac_utils.apply_warp(img_contact, map_x, map_y)
    rect_contact_crop, (crop_x0, crop_y0) = tac_utils.crop_rectified(rect_contact_full, meta)
    rect_contact_inp_full = tac_utils.inpaint_rectified(rect_contact_full, meta)
    rect_contact_inp_crop, _ = tac_utils.crop_rectified(rect_contact_inp_full, meta)
    print(f"  矫正裁剪后 contact 尺寸: {rect_contact_crop.shape}")

    # 将矫正后坐标转换到裁剪后坐标系
    grid_contact_rect_crop = grid_contact_rect.copy()
    grid_contact_rect_crop[:, :, 0] -= crop_x0
    grid_contact_rect_crop[:, :, 1] -= crop_y0
    grid_ref_rect_theory_crop = grid_ref_rect_theory.copy()
    grid_ref_rect_theory_crop[:, :, 0] -= crop_x0
    grid_ref_rect_theory_crop[:, :, 1] -= crop_y0

    print("=" * 60)
    print("Step 6: 可视化")
    print("=" * 60)

    # === 图1: 映射验证 ===
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 7))
    fig1.suptitle(f"Warp 正向点映射验证\nContact: {CONTACT_FILENAME}", fontsize=14)

    ax = axes1[0]
    ax.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    pts_raw = grid_ref_raw[~np.isnan(grid_ref_raw[:, :, 0])]
    ax.scatter(pts_raw[:, 0], pts_raw[:, 1], c='red', s=3, alpha=0.5)
    ax.set_title("原始图像上检测的 ref 网格点")

    ax = axes1[1]
    ax.imshow(cv2.cvtColor(rect_contact_crop, cv2.COLOR_BGR2RGB))
    pts_rect_crop = grid_contact_rect_crop[valid_mask]
    pts_ref_crop = grid_ref_rect_theory_crop[valid_mask]
    ax.scatter(pts_ref_crop[:, 0], pts_ref_crop[:, 1], c='blue', s=5, alpha=0.4, marker='+', label='ref (standard grid)')
    ax.scatter(pts_rect_crop[:, 0], pts_rect_crop[:, 1], c='red', s=5, alpha=0.6, label='contact (warped)')
    ax.legend()
    ax.set_title("矫正后坐标: 标准网格(蓝) vs contact映射(红)")

    fig1.tight_layout()
    out_path1 = os.path.join(OUT_DIR, "demo_09_warp_verify.png")
    fig1.savefig(out_path1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  [保存] {out_path1}")

    # === 图2: 位移热力图（物理坐标 mm）===
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle(f"Displacement Heatmap (mm)\nContact: {CONTACT_FILENAME}", fontsize=14)

    x_mm = np.arange(tac_utils.N_COLS) * tac_utils.PITCH_X_MM
    y_mm = np.arange(tac_utils.N_ROWS) * tac_utils.PITCH_Y_MM
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

    # dx 热力图
    ax = axes2[0, 0]
    im = ax.pcolormesh(X_mm, Y_mm, dx_mm, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"dx (X displacement)\nmax |dx|={np.nanmax(np.abs(dx_mm[valid_mask])):.4f} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="mm")

    # dy 热力图
    ax = axes2[0, 1]
    im = ax.pcolormesh(X_mm, Y_mm, dy_mm, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"dy (Y displacement)\nmax |dy|={np.nanmax(np.abs(dy_mm[valid_mask])):.4f} mm")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="mm")

    # |d| 热力图
    ax = axes2[1, 0]
    im = ax.pcolormesh(X_mm, Y_mm, disp_mag, cmap="hot",
                        vmin=0, vmax=vmax, shading="auto")
    ax.set_title(f"|d| (magnitude)\nmax={vmax:.4f} mm (95th pct)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="mm")

    # quiver 箭头图
    ax = axes2[1, 1]
    ax.imshow(cv2.cvtColor(rect_contact_inp_crop, cv2.COLOR_BGR2RGB), alpha=0.4,
              extent=[0, tac_utils.N_COLS*tac_utils.PITCH_X_MM,
                      tac_utils.N_ROWS*tac_utils.PITCH_Y_MM, 0])
    q = ax.quiver(X_mm[valid_mask], Y_mm[valid_mask],
                  dx_mm[valid_mask], dy_mm[valid_mask],
                  disp_mag[valid_mask], cmap="hot", scale=vmax*8, width=0.005)
    ax.set_title("Displacement vector field (quiver)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(q, ax=ax, label="|d| (mm)")

    fig2.tight_layout()
    out_path2 = os.path.join(OUT_DIR, "demo_09_displacement_heatmap.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [保存] {out_path2}")

    # === 图3: 位移叠加到图像上（裁剪后像素坐标）===
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 7))
    fig3.suptitle(f"Displacement Overlay\nContact: {CONTACT_FILENAME}", fontsize=14)

    ax = axes3[0]
    ax.imshow(cv2.cvtColor(rect_contact_inp_crop, cv2.COLOR_BGR2RGB), alpha=0.7)
    im = ax.scatter(grid_contact_rect_crop[valid_mask, 0], grid_contact_rect_crop[valid_mask, 1],
                    c=dx_mm[valid_mask], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=20, alpha=0.8)
    ax.set_title("dx (blue=left, red=right)")
    ax.set_xlim(0, rect_contact_inp_crop.shape[1])
    ax.set_ylim(rect_contact_inp_crop.shape[0], 0)
    plt.colorbar(im, ax=ax, label="mm", fraction=0.046)

    ax = axes3[1]
    ax.imshow(cv2.cvtColor(rect_contact_inp_crop, cv2.COLOR_BGR2RGB), alpha=0.7)
    im = ax.scatter(grid_contact_rect_crop[valid_mask, 0], grid_contact_rect_crop[valid_mask, 1],
                    c=dy_mm[valid_mask], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    s=20, alpha=0.8)
    ax.set_title("dy (blue=up, red=down)")
    ax.set_xlim(0, rect_contact_inp_crop.shape[1])
    ax.set_ylim(rect_contact_inp_crop.shape[0], 0)
    plt.colorbar(im, ax=ax, label="mm", fraction=0.046)

    ax = axes3[2]
    ax.imshow(cv2.cvtColor(rect_contact_inp_crop, cv2.COLOR_BGR2RGB), alpha=0.7)
    im = ax.scatter(grid_contact_rect_crop[valid_mask, 0], grid_contact_rect_crop[valid_mask, 1],
                    c=disp_mag[valid_mask], cmap="hot", vmin=0, vmax=vmax,
                    s=20, alpha=0.8)
    ax.set_title("|d| (bright=large displacement)")
    ax.set_xlim(0, rect_contact_inp_crop.shape[1])
    ax.set_ylim(rect_contact_inp_crop.shape[0], 0)
    plt.colorbar(im, ax=ax, label="mm", fraction=0.046)

    fig3.tight_layout()
    out_path3 = os.path.join(OUT_DIR, "demo_09_displacement_overlay.png")
    fig3.savefig(out_path3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  [保存] {out_path3}")

    print("=" * 60)
    print("【关键理解】")
    print("  1. 不重新检测！直接用 Phase 1 验证过的原始图像检测结果")
    print("  2. 通过三角剖分逆仿射变换将点正向映射到矫正坐标")
    print("  3. ref 经 warp 映射后应精确落在标准网格位置（误差<0.1px）")
    print("  4. 位移 = contact映射位置 - 标准网格位置，物理意义明确")
    print("  5. 矫正后坐标系 mm/px 均匀，直接乘即可得到物理位移")
    print()
    print("✅ demo_09 完成！")
    print("  下一步: demo_10_batch_validate.py — 10张接触图像批量验证")


if __name__ == "__main__":
    main()
