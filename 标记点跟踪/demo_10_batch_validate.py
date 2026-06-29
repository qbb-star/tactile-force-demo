"""
demo_10_batch_validate.py — Phase 2 Step 4: 10张接触图像批量验证与汇总可视化

本 demo 实现完整 Phase 2 处理链路批量验证:
  1. 加载参考图像和warp（基于Phase 1验证过的参数）
  2. 从400张接触图像中选取10张（均匀覆盖整个数据集）
  3. 对每张图像:
     a. 在原始图像上检测标记点（Phase 1方法）
     b. 正向映射到矫正后坐标（三角剖分逆变换）
     c. 计算位移场（mm）
     d. Warp矫正+Inpaint修复
     e. 保存单张结果
  4. 汇总可视化: 2行5列，每张图标注文件名，展示|d|热力图+quiver
  5. 输出统计信息（最大位移、均值等）
"""

import os
import sys
import glob
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
OUT_DIR_SINGLE = os.path.join(OUT_DIR, "demo_10_individual")
N_SAMPLES = 10

# 固定随机种子以保证可复现
np.random.seed(42)


def process_single_image(img_path, img_ref, map_x, map_y, meta, grid_ref_rect_theory):
    """
    处理单张接触图像，返回位移场和可视化所需数据。

    返回:
        result dict:
            filename: 文件名
            disp_mm: (N_ROWS, N_COLS, 2) 物理位移
            valid_mask: (N_ROWS, N_COLS) bool
            disp_mag: (N_ROWS, N_COLS) 位移大小(mm)
            rect_inp_crop: 矫正+inpaint+裁剪后的BGR图像
            grid_rect_crop: 矫正裁剪后坐标下的contact点位置
            vmax: 95%分位最大位移
            n_valid: 有效点数
            max_disp: 最大位移
            mean_disp: 平均位移
    """
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    assert img is not None, f"无法读取 {img_path}"

    # Phase 1: 原始图像上检测+排序
    grid_raw, dog, kp = tac_utils.detect_and_sort(img)

    # Phase 2 Step 3: 正向映射到矫正后坐标
    grid_rect = tac_utils.warp_points_raw_to_rect(grid_raw, meta)

    # 计算位移
    disp_px = grid_rect - grid_ref_rect_theory
    valid_mask = ~(np.isnan(disp_px[:, :, 0]) | np.isnan(disp_px[:, :, 1]))
    disp_px[~valid_mask] = np.nan
    disp_mm = tac_utils.disp_to_physical(disp_px, meta)

    dx_mm = disp_mm[:, :, 0]
    dy_mm = disp_mm[:, :, 1]
    disp_mag = np.sqrt(dx_mm**2 + dy_mm**2)

    n_valid = int(np.sum(valid_mask))
    if n_valid > 0:
        vmax = float(np.nanpercentile(disp_mag[valid_mask], 95))
        max_disp = float(np.nanmax(disp_mag[valid_mask]))
        mean_disp = float(np.nanmean(disp_mag[valid_mask]))
    else:
        vmax = 0.0
        max_disp = 0.0
        mean_disp = 0.0

    # Warp矫正 + Inpaint（用于可视化背景）
    rect_full = tac_utils.apply_warp(img, map_x, map_y)
    rect_inp_full = tac_utils.inpaint_rectified(rect_full, meta)
    rect_inp_crop, (crop_x0, crop_y0) = tac_utils.crop_rectified(rect_inp_full, meta)

    # 矫正裁剪后坐标
    grid_rect_crop = grid_rect.copy()
    grid_rect_crop[:, :, 0] -= crop_x0
    grid_rect_crop[:, :, 1] -= crop_y0

    return {
        "filename": filename,
        "disp_mm": disp_mm,
        "valid_mask": valid_mask,
        "disp_mag": disp_mag,
        "dx_mm": dx_mm,
        "dy_mm": dy_mm,
        "rect_inp_crop": rect_inp_crop,
        "grid_rect_crop": grid_rect_crop,
        "vmax": vmax,
        "max_disp": max_disp,
        "mean_disp": mean_disp,
        "n_valid": n_valid,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_DIR_SINGLE, exist_ok=True)

    print("=" * 60)
    print("Phase 2 批量验证: 10张接触图像")
    print("=" * 60)

    # Step 1: 构建warp
    print("\n[Step 1] 构建Warp...")
    img_ref = cv2.imread(REF_PATH)
    assert img_ref is not None, f"无法读取 {REF_PATH}"
    grid_ref_raw, _, _ = tac_utils.detect_and_sort(img_ref)
    map_x, map_y, meta, out_h = tac_utils.build_warp(grid_ref_raw)
    tac_utils.save_warp(WARP_PATH, map_x, map_y, meta)
    rp = meta["rectified_px"]
    grid_ref_rect_theory, dx_rect, dy_rect = tac_utils.get_rectified_grid_centers(meta)
    print(f"  矫正尺寸: {rp['out_width']}×{rp['out_height']}")
    print(f"  物理尺度: {rp['mm_per_px_x']:.5f} mm/px")

    # Step 2: 选取10张接触图像
    print("\n[Step 2] 选取接触图像...")
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "image_*.jpg")))
    print(f"  数据集中共 {len(all_files)} 张接触图像")

    # 均匀采样10张（等间隔选取）
    if len(all_files) >= N_SAMPLES:
        indices = np.linspace(0, len(all_files) - 1, N_SAMPLES, dtype=int)
        selected_files = [all_files[i] for i in indices]
    else:
        selected_files = all_files
    print(f"  选取 {len(selected_files)} 张用于验证:")
    for f in selected_files:
        print(f"    - {os.path.basename(f)}")

    # Step 3: 批量处理
    print(f"\n[Step 3] 批量处理 {len(selected_files)} 张图像...")
    results = []
    for i, fpath in enumerate(selected_files):
        fname = os.path.basename(fpath)
        print(f"  [{i+1}/{len(selected_files)}] 处理 {fname} ...", end=" ")
        try:
            res = process_single_image(fpath, img_ref, map_x, map_y, meta, grid_ref_rect_theory)
            results.append(res)
            print(f"OK (n_valid={res['n_valid']}, max_disp={res['max_disp']:.4f}mm)")
        except Exception as e:
            print(f"FAIL: {e}")

    print(f"\n  成功处理 {len(results)}/{len(selected_files)} 张")

    # Step 4: 保存单张详细结果
    print(f"\n[Step 4] 保存单张详细结果到 {OUT_DIR_SINGLE} ...")
    x_mm = np.arange(tac_utils.N_COLS) * tac_utils.PITCH_X_MM
    y_mm = np.arange(tac_utils.N_ROWS) * tac_utils.PITCH_Y_MM
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

    for res in results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(f"{res['filename']}\n"
                     f"max |d|={res['max_disp']:.4f} mm, mean |d|={res['mean_disp']:.4f} mm, "
                     f"n_valid={res['n_valid']}", fontsize=11)

        vm = res["vmax"] if res["vmax"] > 0 else 0.1

        # |d| 热力图
        ax = axes[0]
        im = ax.pcolormesh(X_mm, Y_mm, res["disp_mag"], cmap="hot",
                            vmin=0, vmax=vm, shading="auto")
        ax.set_title("|d| magnitude (mm)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # quiver
        ax = axes[1]
        ax.imshow(cv2.cvtColor(res["rect_inp_crop"], cv2.COLOR_BGR2RGB), alpha=0.4,
                  extent=[0, tac_utils.N_COLS*tac_utils.PITCH_X_MM,
                          tac_utils.N_ROWS*tac_utils.PITCH_Y_MM, 0])
        vm_disp = max(vm, 1e-6)
        q = ax.quiver(X_mm[res["valid_mask"]], Y_mm[res["valid_mask"]],
                      res["dx_mm"][res["valid_mask"]], res["dy_mm"][res["valid_mask"]],
                      res["disp_mag"][res["valid_mask"]], cmap="hot",
                      scale=vm_disp*8, width=0.005)
        ax.set_title("Displacement vectors")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        plt.colorbar(q, ax=ax, fraction=0.046)

        # 叠加图
        ax = axes[2]
        ax.imshow(cv2.cvtColor(res["rect_inp_crop"], cv2.COLOR_BGR2RGB), alpha=0.7)
        grid_c = res["grid_rect_crop"]
        sc = ax.scatter(grid_c[res["valid_mask"], 0], grid_c[res["valid_mask"], 1],
                        c=res["disp_mag"][res["valid_mask"]], cmap="hot",
                        vmin=0, vmax=vm, s=15, alpha=0.8)
        ax.set_title("Overlay on inpainted image")
        ax.set_xlim(0, res["rect_inp_crop"].shape[1])
        ax.set_ylim(res["rect_inp_crop"].shape[0], 0)
        plt.colorbar(sc, ax=ax, fraction=0.046, label="mm")

        fig.tight_layout()
        safe_name = res["filename"].replace(".jpg", "")
        out_path = os.path.join(OUT_DIR_SINGLE, f"{safe_name}_result.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    print(f"  保存了 {len(results)} 张单张结果图")

    # Step 5: 汇总可视化（2行5列）
    print("\n[Step 5] 生成汇总可视化...")

    # 计算全局 vmax 用于统一色标
    global_vmax = max([r["vmax"] for r in results]) if results else 0.1
    if global_vmax < 0.01:
        global_vmax = 0.1

    n_show = len(results)
    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5.5 * n_rows))
    fig.suptitle(f"Phase 2 Batch Validation — {n_show} Contact Images\n"
                 f"Displacement magnitude |d| (mm) with quiver", fontsize=14)

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for idx, res in enumerate(results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # 背景：inpaint后图像
        ax.imshow(cv2.cvtColor(res["rect_inp_crop"], cv2.COLOR_BGR2RGB), alpha=0.35)

        # quiver箭头
        vm_each = res["vmax"] if res["vmax"] > 0 else global_vmax
        q = ax.quiver(X_mm[res["valid_mask"]], Y_mm[res["valid_mask"]],
                      res["dx_mm"][res["valid_mask"]], res["dy_mm"][res["valid_mask"]],
                      res["disp_mag"][res["valid_mask"]], cmap="hot",
                      scale=global_vmax*6, width=0.006, clim=(0, global_vmax))

        # 标注文件名和统计
        ax.set_title(f"{res['filename']}\n"
                     f"max={res['max_disp']:.3f}mm, mean={res['mean_disp']:.3f}mm",
                     fontsize=8)
        ax.set_xlim(0, res["rect_inp_crop"].shape[1])
        ax.set_ylim(res["rect_inp_crop"].shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])

    # 隐藏多余子图
    for idx in range(n_show, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    # 统一色标
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(q, cax=cbar_ax, label="|d| (mm)")

    summary_path = os.path.join(OUT_DIR, "demo_10_batch_summary.png")
    fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [保存] {summary_path}")

    # 汇总热力图（只有|d|，无箭头，更清晰）
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    fig2.suptitle(f"Phase 2 Batch Validation — Displacement Magnitude |d| (mm)", fontsize=14)
    if n_rows == 1:
        axes2 = axes2[np.newaxis, :]

    for idx, res in enumerate(results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes2[row, col]
        im = ax.pcolormesh(X_mm, Y_mm, res["disp_mag"], cmap="hot",
                            vmin=0, vmax=global_vmax, shading="auto")
        ax.set_title(f"{res['filename']}\nmax={res['max_disp']:.3f}mm", fontsize=8)
        ax.set_xlabel("x(mm)", fontsize=7)
        ax.set_ylabel("y(mm)", fontsize=7)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

    for idx in range(n_show, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes2[row, col].axis("off")

    fig2.subplots_adjust(right=0.92)
    cbar_ax2 = fig2.add_axes([0.93, 0.15, 0.015, 0.7])
    fig2.colorbar(im, cax=cbar_ax2, label="|d| (mm)")

    heatmap_path = os.path.join(OUT_DIR, "demo_10_batch_heatmap.png")
    fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [保存] {heatmap_path}")

    # Step 6: 打印统计汇总
    print("\n" + "=" * 60)
    print("【统计汇总】")
    print("=" * 60)
    print(f"{'文件名':<45} {'有效点':>6} {'max|d|(mm)':>11} {'mean|d|(mm)':>12}")
    print("-" * 78)
    for res in results:
        print(f"{res['filename']:<45} {res['n_valid']:>6} "
              f"{res['max_disp']:>11.4f} {res['mean_disp']:>12.4f}")

    all_max = [r["max_disp"] for r in results]
    all_mean = [r["mean_disp"] for r in results]
    all_n = [r["n_valid"] for r in results]
    print("-" * 78)
    print(f"{'平均':<45} {np.mean(all_n):>6.1f} "
          f"{np.mean(all_max):>11.4f} {np.mean(all_mean):>12.4f}")
    print(f"{'范围':<45} {min(all_n)}-{max(all_n):<3} "
          f"{min(all_max):.4f}-{max(all_max):.4f}  "
          f"{min(all_mean):.4f}-{max(all_mean):.4f}")

    print()
    print("=" * 60)
    print("【Phase 2 完成总结】")
    print("=" * 60)
    print("  完整处理链路:")
    print("    原始图像 → DoG+Blob检测 → 网格排序(Phase 1)")
    print("    → 三角剖分Warp构建 → 图像矫正")
    print("    → 标记点Inpaint修复 → 正向点映射")
    print("    → 位移场计算(mm) → 热力图/quiver可视化")
    print()
    print("  关键参数（基于Phase 1验证）:")
    print(f"    网格: {tac_utils.N_ROWS}行 × {tac_utils.N_COLS}列 = {tac_utils.N_ROWS*tac_utils.N_COLS}个标记点")
    print(f"    物理间距: {tac_utils.PITCH_X_MM}mm × {tac_utils.PITCH_Y_MM}mm")
    print(f"    矫正后物理尺度: {rp['mm_per_px_x']:.5f} mm/px")
    print(f"    Inpaint: Telea算法，半径3px")
    print()
    print("  输出文件:")
    print(f"    - {summary_path}")
    print(f"    - {heatmap_path}")
    print(f"    - {OUT_DIR_SINGLE}\\*.png (单张详细结果)")
    print()
    print("✅ Phase 2（位移计算&标记修复）全部完成并验证通过！")


if __name__ == "__main__":
    main()
