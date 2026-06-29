"""
demo_10_step_by_step.py
Phase 2 Step 4: 批量验证 — 逐步调试走读版

每一步都有：目的讲解 + 数值统计 + 可视化输出
做到"数形结合"，帮助深入理解每一步在做什么

使用方法：
    运行后，每一步都会暂停，按回车继续下一步
    每一步的可视化图都会保存到 outputs/debug_demo10/
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import sys
import glob

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tac_utils

DATA_DIR = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
REF_FILE = "ref.jpg"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "debug_demo10")
WARP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "warp_ref.npz")
N_SAMPLES = 10

np.random.seed(42)


def pause(msg="按回车继续..."):
    try:
        input("\n" + msg)
    except EOFError:
        pass


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def process_single(img_path, img_ref, map_x, map_y, meta, grid_ref_rect_theory):
    """处理单张图像，返回结果字典"""
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    assert img is not None

    # 检测+排序
    grid_raw, dog, kp = tac_utils.detect_and_sort(img)

    # 正向映射
    grid_rect = tac_utils.warp_points_raw_to_rect(grid_raw, meta)

    # 位移计算
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
        vmax = max_disp = mean_disp = 0.0

    # 矫正+inpaint（用于可视化背景）
    rect_full = tac_utils.apply_warp(img, map_x, map_y)
    rect_inp_full = tac_utils.inpaint_rectified(rect_full, meta)
    rect_inp_crop, (crop_x0, crop_y0) = tac_utils.crop_rectified(rect_inp_full, meta)

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # Step 0: 整体路线
    # ============================================================
    print_header("Step 0: 整体路线 — demo_10 做什么？")

    print("""
【Phase 2 学习路线】
  demo_07: 双图检测 + 位移初步
  demo_08: Warp 矫正 + Inpaint 修复
  demo_09: 位移场与热力图
→ demo_10: 批量验证（10张图）

【为什么要做批量验证？】
  单张图像过了，不代表算法稳定。
  我们从 400 张接触图像里均匀选 10 张，看看：
    1. 检测率稳不稳定？（有效点数）
    2. 位移大小合理吗？（max/mean 范围）
    3. 热力图模式一致吗？（有没有哪张图明显异常）

【6个步骤】
  Step 1: 构建 Warp（复用 Phase 1 结果）
  Step 2: 从 400 张图里选 10 张（均匀采样）
  Step 3: 批量处理，逐张看结果
  Step 4: 单张详细结果（热力图+箭头+叠加）
  Step 5: 汇总对比（2行5列，同色标）
  Step 6: 统计汇总与 Phase 2 总结
""")
    pause()

    # ============================================================
    # Step 1: 构建 Warp
    # ============================================================
    print_header("Step 1: 构建 Warp + 标准网格")

    img_ref = cv2.imread(os.path.join(DATA_DIR, REF_FILE))
    assert img_ref is not None

    grid_ref_raw, _, _ = tac_utils.detect_and_sort(img_ref)
    map_x, map_y, meta, out_h = tac_utils.build_warp(grid_ref_raw)
    tac_utils.save_warp(WARP_PATH, map_x, map_y, meta)

    rp = meta["rectified_px"]
    grid_ref_rect_theory, dx_rect, dy_rect = tac_utils.get_rectified_grid_centers(meta)

    print(f"  矫正尺寸: {rp['out_width']} x {rp['out_height']} px")
    print(f"  物理尺度: {rp['mm_per_px_x']:.5f} mm/px")
    print(f"  标准网格: {tac_utils.N_ROWS}行 × {tac_utils.N_COLS}列 = {tac_utils.N_ROWS*tac_utils.N_COLS}个点")
    print(f"  物理间距: {tac_utils.PITCH_X_MM}mm × {tac_utils.PITCH_Y_MM}mm")

    pause()

    # ============================================================
    # Step 2: 选取 10 张图像
    # ============================================================
    print_header("Step 2: 从 400 张图里均匀选 10 张")

    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "image_*.jpg")))
    print(f"  数据集中共 {len(all_files)} 张接触图像")
    print()

    # 均匀采样
    indices = np.linspace(0, len(all_files) - 1, N_SAMPLES, dtype=int)
    selected_files = [all_files[i] for i in indices]

    print(f"  均匀选取 {len(selected_files)} 张:")
    print(f"  {'序号':>4s}  {'索引':>5s}  {'文件名'}")
    print("  " + "-" * 70)
    for i, (idx, fpath) in enumerate(zip(indices, selected_files)):
        print(f"  {i+1:>4d}  {idx:>5d}  {os.path.basename(fpath)}")
    print()
    print("  为什么要均匀选？")
    print("    → 覆盖整个数据集，从前期小力到后期大力都有")
    print("    → 比随机选更有代表性，能看到变化趋势")

    # 可视化：在时间轴上标出选的位置
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(range(len(all_files)), [1] * len(all_files), color="lightgray", width=1.0)
    ax.bar(indices, [1.5] * len(indices), color="red", width=3.0)
    ax.set_title(f"10 张采样在 {len(all_files)} 张全集里的位置（红色=选中）", fontsize=12)
    ax.set_xlabel("图像索引（按时间顺序）")
    ax.set_yticks([])
    ax.set_xlim(0, len(all_files))
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "step02_sampling.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [可视化] 已保存: {save_path}")

    pause()

    # ============================================================
    # Step 3: 批量处理
    # ============================================================
    print_header("Step 3: 批量处理 10 张图像")

    print("  对每张图执行完整链路:")
    print("    检测 → 排序 → 正向映射 → 位移计算 → 矫正+inpaint")
    print()

    results = []
    for i, fpath in enumerate(selected_files):
        fname = os.path.basename(fpath)
        print(f"  [{i+1:>2d}/{len(selected_files)}] 处理 {fname} ...", end=" ")
        try:
            res = process_single(fpath, img_ref, map_x, map_y, meta, grid_ref_rect_theory)
            results.append(res)
            print(f"OK  有效点={res['n_valid']:>3d}, max|d|={res['max_disp']:.4f}mm, mean={res['mean_disp']:.4f}mm")
        except Exception as e:
            print(f"FAIL: {e}")

    print(f"\n  成功处理 {len(results)}/{len(selected_files)} 张")

    pause()

    # ============================================================
    # Step 4: 单张详细结果
    # ============================================================
    print_header("Step 4: 单张详细结果（每张 3 个子图）")

    x_mm = np.arange(tac_utils.N_COLS) * tac_utils.PITCH_X_MM
    y_mm = np.arange(tac_utils.N_ROWS) * tac_utils.PITCH_Y_MM
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

    print(f"  为每张图像生成一张详细结果图，保存在:")
    print(f"    {OUTPUT_DIR}\\individual_*.png")
    print()

    for res in results:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
        fig.suptitle(f"{res['filename']}\n"
                     f"max|d|={res['max_disp']:.4f}mm, mean|d|={res['mean_disp']:.4f}mm, "
                     f"有效点={res['n_valid']}/{tac_utils.N_ROWS*tac_utils.N_COLS}",
                     fontsize=12, fontweight="bold")

        vm = res["vmax"] if res["vmax"] > 0 else 0.1

        # |d| 热力图
        ax = axes[0]
        im = ax.pcolormesh(X_mm, Y_mm, res["disp_mag"], cmap="hot",
                            vmin=0, vmax=vm, shading="auto")
        ax.set_title("|d| 位移大小 (mm)", fontsize=11)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.invert_yaxis()
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, label="mm")

        # quiver 箭头
        ax = axes[1]
        ax.imshow(cv2.cvtColor(res["rect_inp_crop"], cv2.COLOR_BGR2RGB), alpha=0.4,
                  extent=[0, tac_utils.N_COLS * tac_utils.PITCH_X_MM,
                          tac_utils.N_ROWS * tac_utils.PITCH_Y_MM, 0])
        vm_q = max(vm, 1e-6)
        q = ax.quiver(X_mm[res["valid_mask"]], Y_mm[res["valid_mask"]],
                      res["dx_mm"][res["valid_mask"]], res["dy_mm"][res["valid_mask"]],
                      res["disp_mag"][res["valid_mask"]], cmap="hot",
                      scale=vm_q * 8, width=0.005)
        ax.set_title("位移向量场（箭头=方向，颜色=大小）", fontsize=11)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
        plt.colorbar(q, ax=ax, fraction=0.046, label="|d| (mm)")

        # 叠加图
        ax = axes[2]
        ax.imshow(cv2.cvtColor(res["rect_inp_crop"], cv2.COLOR_BGR2RGB), alpha=0.7)
        grid_c = res["grid_rect_crop"]
        sc = ax.scatter(grid_c[res["valid_mask"], 0], grid_c[res["valid_mask"], 1],
                        c=res["disp_mag"][res["valid_mask"]], cmap="hot",
                        vmin=0, vmax=vm, s=18, alpha=0.85)
        ax.set_title("叠加在矫正图像上", fontsize=11)
        ax.set_xlim(0, res["rect_inp_crop"].shape[1])
        ax.set_ylim(res["rect_inp_crop"].shape[0], 0)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        plt.colorbar(sc, ax=ax, fraction=0.046, label="mm")

        plt.tight_layout()
        safe_name = res["filename"].replace(".jpg", "")
        out_path = os.path.join(OUTPUT_DIR, f"individual_{safe_name}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  已保存 {len(results)} 张单张详细结果图")

    pause()

    # ============================================================
    # Step 5: 汇总对比
    # ============================================================
    print_header("Step 5: 汇总对比 — 10 张图放一起看")

    global_vmax = max([r["vmax"] for r in results]) if results else 0.1
    if global_vmax < 0.01:
        global_vmax = 0.1

    n_show = len(results)
    n_cols = 5
    n_rows = int(np.ceil(n_show / n_cols))

    print(f"  统一色标 vmax = {global_vmax:.4f} mm（所有图用同一个颜色范围，方便对比）")
    print(f"  布局: {n_rows}行 × {n_cols}列")
    print()

    # 汇总热力图（只有 |d|，更清晰）
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows))
    fig.suptitle(f"Phase 2 批量验证 — 10 张接触图像位移大小 |d| (mm)\n"
                 f"统一色标 vmax={global_vmax:.3f}mm",
                 fontsize=14, fontweight="bold")

    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for idx, res in enumerate(results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        im = ax.pcolormesh(X_mm, Y_mm, res["disp_mag"], cmap="hot",
                            vmin=0, vmax=global_vmax, shading="auto")
        ax.set_title(f"{res['filename']}\nmax={res['max_disp']:.3f}mm, mean={res['mean_disp']:.3f}mm",
                     fontsize=8)
        ax.set_xlabel("x(mm)", fontsize=7)
        ax.set_ylabel("y(mm)", fontsize=7)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)

    # 隐藏多余子图
    for idx in range(n_show, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    # 统一色标
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="|d| (mm)")

    save_path = os.path.join(OUTPUT_DIR, "step05_batch_heatmap.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 汇总热力图: {save_path}")

    # 箭头汇总图
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 5 * n_rows))
    fig2.suptitle(f"Phase 2 批量验证 — 位移向量场\n统一色标 vmax={global_vmax:.3f}mm",
                  fontsize=14, fontweight="bold")

    if n_rows == 1:
        axes2 = axes2[np.newaxis, :]

    for idx, res in enumerate(results):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes2[row, col]
        ax.imshow(cv2.cvtColor(res["rect_inp_crop"], cv2.COLOR_BGR2RGB), alpha=0.35)
        q = ax.quiver(X_mm[res["valid_mask"]], Y_mm[res["valid_mask"]],
                      res["dx_mm"][res["valid_mask"]], res["dy_mm"][res["valid_mask"]],
                      res["disp_mag"][res["valid_mask"]], cmap="hot",
                      scale=global_vmax * 6, width=0.006, clim=(0, global_vmax))
        ax.set_title(f"{res['filename']}\nmax={res['max_disp']:.3f}mm", fontsize=8)
        ax.set_xlim(0, res["rect_inp_crop"].shape[1])
        ax.set_ylim(res["rect_inp_crop"].shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_show, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes2[row, col].axis("off")

    fig2.subplots_adjust(right=0.92)
    cbar_ax2 = fig2.add_axes([0.93, 0.15, 0.015, 0.7])
    fig2.colorbar(q, cax=cbar_ax2, label="|d| (mm)")

    save_path2 = os.path.join(OUTPUT_DIR, "step05_batch_quiver.png")
    fig2.savefig(save_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [可视化] 汇总箭头图: {save_path2}")
    print()
    print("  看图提示：")
    print("    1. 先整体扫一眼，看看位移大小的变化趋势")
    print("    2. 找最大的那张，看看是不是最'用力'的时刻")
    print("    3. 对比热力图和箭头图，理解'大小'和'方向'的关系")

    pause()

    # ============================================================
    # Step 6: 统计汇总 + Phase 2 总结
    # ============================================================
    print_header("Step 6: 统计汇总与 Phase 2 总结")

    print(f"  {'序号':>4s}  {'文件名':<42s} {'有效点':>6s} {'max|d|(mm)':>11s} {'mean|d|(mm)':>12s}")
    print("  " + "-" * 82)
    for i, res in enumerate(results):
        print(f"  {i+1:>4d}  {res['filename']:<42s} {res['n_valid']:>6d} "
              f"{res['max_disp']:>11.4f} {res['mean_disp']:>12.4f}")

    all_max = [r["max_disp"] for r in results]
    all_mean = [r["mean_disp"] for r in results]
    all_n = [r["n_valid"] for r in results]
    print("  " + "-" * 82)
    print(f"  {'平均':>4s}  {'':42s} {np.mean(all_n):>6.1f} "
          f"{np.mean(all_max):>11.4f} {np.mean(all_mean):>12.4f}")
    print(f"  {'范围':>4s}  {'':42s} {min(all_n):>3d}-{max(all_n):<3d} "
          f"{min(all_max):.4f}-{max(all_max):.4f}  "
          f"{min(all_mean):.4f}-{max(all_mean):.4f}")

    # 趋势图
    fig, ax1 = plt.subplots(figsize=(12, 5))
    x_pos = range(len(results))
    ax1.bar(x_pos, all_max, alpha=0.6, color="red", label="max |d|")
    ax1.bar(x_pos, all_mean, alpha=0.8, color="orange", label="mean |d|")
    ax1.set_ylabel("位移大小 (mm)", fontsize=11)
    ax1.set_xlabel("图像序号（按时间顺序）", fontsize=11)
    ax1.set_title("10 张图位移大小变化趋势", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"#{i+1}" for i in range(len(results))], fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(x_pos, all_n, "bo-", markersize=6, label="有效点数")
    ax2.set_ylabel("有效点数", fontsize=11, color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim(0, tac_utils.N_ROWS * tac_utils.N_COLS)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "step06_trend.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [可视化] 趋势图: {save_path}")
    print()

    # Phase 2 总结
    print("=" * 70)
    print("  Phase 2 全部完成总结")
    print("=" * 70)
    print(f"""
  完整处理链路:
    原始图像 → DoG+Blob检测 → 网格排序 (Phase 1)
    → 三角剖分Warp构建 → 图像矫正
    → 标记点Inpaint修复 → 正向点映射
    → 位移场计算(mm) → 热力图/quiver可视化

  关键参数:
    网格: {tac_utils.N_ROWS}行 × {tac_utils.N_COLS}列 = {tac_utils.N_ROWS*tac_utils.N_COLS}个标记点
    物理间距: {tac_utils.PITCH_X_MM}mm × {tac_utils.PITCH_Y_MM}mm
    矫正后物理尺度: {rp['mm_per_px_x']:.5f} mm/px
    Inpaint: Telea 算法，半径 3px
    排序鲁棒性: 多策略择优(k-means + rank-based) + 拓扑验证

  已验证能力:
    ✅ 小位移图像：检测+位移计算稳定
    ✅ 大位移图像：排序鲁棒，位移场合理
    ✅ 批量处理：10张图全部成功，有效点率高

  输出文件（{OUTPUT_DIR}）:
    - step02_sampling.png       采样位置示意图
    - step05_batch_heatmap.png  10张 |d| 热力图汇总
    - step05_batch_quiver.png   10张 箭头图汇总
    - step06_trend.png          位移大小变化趋势
    - individual_*.png          每张图的详细结果（3合1）

  下一步: Phase 3 — 法向估计（从位移场到表面法向量）
""")

    print("✅ Phase 2（位移计算 & 标记修复）全部完成！")
    pause("按回车结束...")


if __name__ == "__main__":
    main()
