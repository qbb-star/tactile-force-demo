"""
demo_09_step_by_step.py
Phase 2 Step 3: 位移场计算与热力图 — 逐步调试走读版

每一步都有：目的讲解 + 数值统计 + 可视化输出
做到"数形结合"，帮助深入理解每一步在做什么

使用方法：
    运行后，每一步都会暂停，按回车继续下一步
    每一步的可视化图都会保存到 outputs/debug_demo09/
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tac_utils

DATA_DIR = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
REF_FILE = "ref.jpg"
CONTACT_FILE = "image_20260430_135445_239.jpg"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "debug_demo09")
WARP_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "warp_ref.npz")


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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ============================================================
    # Step 0: 整体路线总览
    # ============================================================
    print_header("Step 0: 整体路线总览 — 我们今天要学什么？")

    print("""
【Phase 2 学习路线】
  demo_07: 双图检测 + 位移初步（原始图像上算位移）
  demo_08: Warp 矫正 + Inpaint 标记点修复
→ demo_09: 位移场计算与热力图（矫正后坐标，物理单位 mm）
  demo_10: 批量验证（10张图）

【demo_09 核心解决的问题】
  Q: 为什么不在原始图像上直接算位移？
  A: 因为原始图像有透视畸变，上下左右的 mm/px 不一样，
     位移的"物理意义"不明确。

  Q: 那怎么办？
  A: 把检测到的标记点"搬"到矫正后的坐标里，再算位移。
     矫正后坐标下 mm/px 是均匀的，位移直接有物理意义。

【关键新概念：正向点映射】
  Warp 的 map_x/map_y 是"反向映射"（矫正后→原始）
  但我们需要"正向映射"（原始→矫正后）
  怎么实现？ → 用三角剖分的逆变换！

  类比：
    map_x/map_y = "矫正图的每个像素，去原图哪里取色"（反向查表）
    正向点映射 = "原图的一个点，搬到矫正图的哪个位置"（正向搬家）

【7个步骤】
  Step 1: 构建 Warp（含三角剖分）
  Step 2: 原始图像上检测 contact 网格点
  Step 3: 正向点映射原理可视化（重点！）
  Step 4: 位移场计算（像素 → mm）
  Step 5: 位移热力图（dx, dy, |d|）
  Step 6: 箭头图 + 叠加图
  Step 7: 总结回顾
""")
    pause()

    # ============================================================
    # Step 1: 构建 Warp
    # ============================================================
    print_header("Step 1: 构建 Warp（含三角剖分信息）")

    img_ref = cv2.imread(os.path.join(DATA_DIR, REF_FILE))
    assert img_ref is not None

    print("【做什么？】")
    print("  从 ref.jpg 检测标记点 → 构建 Warp → 保存 npz 文件")
    print("  这次的 Warp 除了 map_x/map_y，还多存了三角剖分信息 tris")
    print()

    grid_ref_raw, dog_ref, kp_ref = tac_utils.detect_and_sort(img_ref)
    n_valid = np.sum(~np.isnan(grid_ref_raw[:, :, 0]))
    print(f"  ref 检测: {len(kp_ref)} 个点，有效网格 {n_valid}/{tac_utils.N_COLS*tac_utils.N_ROWS}")

    map_x, map_y, meta, out_h = tac_utils.build_warp(grid_ref_raw)
    tac_utils.save_warp(WARP_PATH, map_x, map_y, meta)

    rp = meta["rectified_px"]
    print(f"  Warp 尺寸: {rp['out_width']} x {rp['out_height']} px")
    print(f"  三角形数量: {len(meta['tris'])} 个（用于正向点映射）")
    print(f"  物理尺度: {rp['mm_per_px_x']:.5f} mm/px (x方向)")
    print(f"                {rp['mm_per_px_y']:.5f} mm/px (y方向)")
    print()

    # 标准网格（理论位置）
    grid_ref_rect_theory, dx_rect, dy_rect = tac_utils.get_rectified_grid_centers(meta)
    n_std = np.sum(~np.isnan(grid_ref_rect_theory[:, :, 0]))
    print(f"  标准网格: {n_std} 个点（理论上矫正后 ref 应该在这里）")
    print(f"  矫正后行间距 dy_rect = {dy_rect:.2f} px")
    print(f"  矫正后列间距 dx_rect = {dx_rect:.2f} px")

    pause()

    # ============================================================
    # Step 2: 原始图像上检测 contact 网格点
    # ============================================================
    print_header("Step 2: 在原始接触图像上检测网格点")

    img_contact = cv2.imread(os.path.join(DATA_DIR, CONTACT_FILE))
    assert img_contact is not None

    print("【做什么？】")
    print("  对 contact 图像用同样的方法检测+排序")
    print("  注意：检测是在原始图像上做的，不是矫正后的！")
    print("  原因：原始图像分辨率高，检测更准确")
    print()

    grid_contact_raw, dog_contact, kp_contact = tac_utils.detect_and_sort(img_contact)
    n_contact = np.sum(~np.isnan(grid_contact_raw[:, :, 0]))
    print(f"  contact 检测: {len(kp_contact)} 个点，有效网格 {n_contact}/{tac_utils.N_COLS*tac_utils.N_ROWS}")
    print()

    # 挑几个点看看原始坐标
    print("  原始坐标示例（挑几个代表性的点）:")
    samples = [(0, 0), (0, 15), (10, 7), (20, 0), (20, 15)]
    for r, c in samples:
        pt = grid_contact_raw[r, c]
        if not np.isnan(pt[0]):
            print(f"    第{r:2d}行 第{c:2d}列: ({pt[0]:7.1f}, {pt[1]:7.1f}) px")

    pause()

    # ============================================================
    # Step 3: 正向点映射 — 重点！
    # ============================================================
    print_header("Step 3: 正向点映射 — 把点从原图搬到矫正图")

    print("""
【核心概念：正向点映射是什么？】

  Warp 矫正图像用的是"反向映射":
    矫正图(X,Y) → 查 map_x[Y,X], map_y[Y,X] → 去原图(x,y)取色

  但我们现在要做的是"正向映射":
    原图的一个点(x,y) → 它在矫正图的哪个位置？

  为什么不能直接用 map_x/map_y 反推？
    因为 map_x/map_y 是"稠密表格"，只能正向查（给X,Y找x,y）
    反向查（给x,y找X,Y）需要解方程，很麻烦。

  解决方案：三角剖分 + 仿射变换
    1. 把矫正后的标准网格点做 Delaunay 三角剖分
    2. 每个三角形在原图上也有对应的三个顶点
    3. 知道一个点在原图某个三角形内，就能用仿射变换算出它在矫正图的位置

  类比：
    就像把一张有三角形网格的橡皮膜拉伸变形
    你知道变形前每个三角形的三个顶点位置
    也知道变形后三个顶点的位置
    那么三角形内任意一点的新位置，都可以用比例关系算出来
""")

    # 执行正向映射
    grid_ref_rect = tac_utils.warp_points_raw_to_rect(grid_ref_raw, meta)
    grid_contact_rect = tac_utils.warp_points_raw_to_rect(grid_contact_raw, meta)

    n_ref_rect = np.sum(~np.isnan(grid_ref_rect[:, :, 0]))
    n_contact_rect = np.sum(~np.isnan(grid_contact_rect[:, :, 0]))
    print(f"  ref 映射后有效点: {n_ref_rect}")
    print(f"  contact 映射后有效点: {n_contact_rect}")
    print()

    # 验证 ref 映射后是否落在标准网格位置
    ref_err = grid_ref_rect - grid_ref_rect_theory
    ref_err_mag = np.sqrt(ref_err[:, :, 0]**2 + ref_err[:, :, 1]**2)
    valid_ref = ~np.isnan(ref_err_mag)
    print("  【精度验证】ref 经 warp 映射后，与标准网格的偏差:")
    print(f"    平均偏差: {np.nanmean(ref_err_mag[valid_ref]):.4f} px")
    print(f"    最大偏差: {np.nanmax(ref_err_mag[valid_ref]):.4f} px")
    print("  （这个值越小越好，说明正向映射很准）")
    print()

    # 挑几个点对比
    print("  坐标对比示例（像素）:")
    print(f"    {'位置':>12s} {'标准网格(x,y)':>22s} {'映射后(x,y)':>22s} {'偏差(px)':>10s}")
    print("    " + "-" * 70)
    for r, c in samples:
        pt_std = grid_ref_rect_theory[r, c]
        pt_map = grid_ref_rect[r, c]
        if not np.isnan(pt_std[0]) and not np.isnan(pt_map[0]):
            err = np.sqrt((pt_std[0]-pt_map[0])**2 + (pt_std[1]-pt_map[1])**2)
            print(f"    第{r:2d}行{c:2d}列  ({pt_std[0]:6.1f},{pt_std[1]:6.1f})  "
                  f"({pt_map[0]:6.1f},{pt_map[1]:6.1f})   {err:.3f}")

    # 可视化：三角剖分 + 映射前后对比
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle("正向点映射原理可视化", fontsize=15, fontweight="bold")

    # 左：原图 ref 网格 + 三角剖分
    ax = axes[0]
    ax.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    pts_raw_valid = grid_ref_raw[valid_ref]
    ax.scatter(pts_raw_valid[:, 0], pts_raw_valid[:, 1], c="lime", s=8, zorder=5)
    # 画三角剖分
    for tri in meta["tris"]:
        tp = tri["tri_raw"]  # 三角形三个顶点的原始坐标 (3, 2)
        ax.plot([tp[0, 0], tp[1, 0], tp[2, 0], tp[0, 0]],
                [tp[0, 1], tp[1, 1], tp[2, 1], tp[0, 1]],
                color="cyan", linewidth=0.5, alpha=0.6)
    ax.set_title("原始图像: ref 网格 + Delaunay 三角剖分", fontsize=12)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    # 中：矫正后标准网格 + 三角剖分
    ax = axes[1]
    ref_rect_rgb = cv2.cvtColor(tac_utils.apply_warp(img_ref, map_x, map_y), cv2.COLOR_BGR2RGB)
    ax.imshow(ref_rect_rgb)
    pts_rect_valid = grid_ref_rect_theory[valid_ref]
    ax.scatter(pts_rect_valid[:, 0], pts_rect_valid[:, 1], c="lime", s=8, zorder=5)
    for tri in meta["tris"]:
        tq = tri["tri_can"]  # 三角形三个顶点的矫正后坐标 (3, 2)
        ax.plot([tq[0, 0], tq[1, 0], tq[2, 0], tq[0, 0]],
                [tq[0, 1], tq[1, 1], tq[2, 1], tq[0, 1]],
                color="cyan", linewidth=0.5, alpha=0.6)
    ax.set_title("矫正后坐标: 标准网格 + 三角剖分", fontsize=12)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    # 右：contact 映射前后散点对比
    ax = axes[2]
    valid_c = ~np.isnan(grid_contact_rect[:, :, 0])
    pts_c_raw = grid_contact_raw[valid_c]
    pts_c_rect = grid_contact_rect[valid_c]
    # 归一化到 0-1 范围方便对比形状
    def normalize(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        xn = (x - x.min()) / (x.max() - x.min())
        yn = (y - y.min()) / (y.max() - y.min())
        return np.column_stack([xn, yn])

    raw_norm = normalize(pts_c_raw)
    rect_norm = normalize(pts_c_rect)
    ax.scatter(raw_norm[:, 0], raw_norm[:, 1], c="red", s=15, alpha=0.6, label="原始坐标（归一化）")
    ax.scatter(rect_norm[:, 0] + 1.2, rect_norm[:, 1], c="blue", s=15, alpha=0.6, label="矫正后坐标（归一化）")
    ax.set_title("形状对比: 原始 vs 矫正后（归一化）", fontsize=12)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "step03_forward_mapping.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [可视化] 已保存: {save_path}")
    print("  看图提示：")
    print("    左图=原图上的网格（透视变形的）")
    print("    中图=矫正后的网格（横平竖直的）")
    print("    右图=归一化后形状对比（看矫正前后网格形状的区别）")

    pause()

    # ============================================================
    # Step 4: 位移场计算
    # ============================================================
    print_header("Step 4: 位移场计算（像素 → 毫米）")

    print("""
【位移怎么算？】
  位移 = contact 矫正后位置 - 标准网格位置（ref 的理论位置）

  注意：为什么减"标准网格"而不是"ref映射后位置"？
    因为 ref 映射后位置和标准网格几乎重合（误差<0.1px）
    用标准网格更"干净"，不受检测噪声影响
    标准网格是数学上完美均匀的网格

【像素 → 毫米】
  矫正后坐标下 mm/px 是均匀的
  dx_mm = dx_px × mm_per_px_x
  dy_mm = dy_px × mm_per_px_y
""")

    disp_px = grid_contact_rect - grid_ref_rect_theory
    valid_mask = ~(np.isnan(disp_px[:, :, 0]) | np.isnan(disp_px[:, :, 1]))
    disp_px[~valid_mask] = np.nan
    disp_mm = tac_utils.disp_to_physical(disp_px, meta)

    dx_mm = disp_mm[:, :, 0]
    dy_mm = disp_mm[:, :, 1]
    disp_mag = np.sqrt(dx_mm**2 + dy_mm**2)

    n_valid = int(np.sum(valid_mask))
    vmax = float(np.nanpercentile(disp_mag[valid_mask], 95))
    print(f"  有效点数: {n_valid}/{tac_utils.N_COLS*tac_utils.N_ROWS}")
    print()
    print("  【位移统计（物理单位 mm）】")
    print(f"    |d|  均值: {np.nanmean(disp_mag[valid_mask]):.4f} mm")
    print(f"    |d|  最大值: {np.nanmax(disp_mag[valid_mask]):.4f} mm")
    print(f"    |d|  95%分位: {vmax:.4f} mm （热力图常用这个当上限）")
    print(f"    dx   范围: [{np.nanmin(dx_mm[valid_mask]):.4f}, {np.nanmax(dx_mm[valid_mask]):.4f}] mm")
    print(f"    dy   范围: [{np.nanmin(dy_mm[valid_mask]):.4f}, {np.nanmax(dy_mm[valid_mask]):.4f}] mm")
    print()

    # 找位移最大的点
    max_idx = np.unravel_index(np.nanargmax(disp_mag), disp_mag.shape)
    print(f"  位移最大的点: 第{max_idx[0]}行 第{max_idx[1]}列")
    print(f"    dx = {dx_mm[max_idx]:.4f} mm")
    print(f"    dy = {dy_mm[max_idx]:.4f} mm")
    print(f"    |d| = {disp_mag[max_idx]:.4f} mm")

    pause()

    # ============================================================
    # Step 5: 位移热力图
    # ============================================================
    print_header("Step 5: 位移热力图可视化（dx, dy, |d|）")

    print("【三种热力图的意义】")
    print("  dx热力图: 水平方向位移（红=向右，蓝=向左）")
    print("  dy热力图: 垂直方向位移（红=向下，蓝=向上）")
    print("  |d|热力图: 位移大小（越亮越大）")
    print()

    x_mm = np.arange(tac_utils.N_COLS) * tac_utils.PITCH_X_MM
    y_mm = np.arange(tac_utils.N_ROWS) * tac_utils.PITCH_Y_MM
    X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"位移热力图（物理单位 mm）\n{CONTACT_FILE}", fontsize=14, fontweight="bold")

    # dx
    ax = axes[0, 0]
    im = ax.pcolormesh(X_mm, Y_mm, dx_mm, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"dx — 水平位移\n(红=向右, 蓝=向左)\nmax |dx|={np.nanmax(np.abs(dx_mm[valid_mask])):.4f} mm", fontsize=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="mm", fraction=0.046)

    # dy
    ax = axes[0, 1]
    im = ax.pcolormesh(X_mm, Y_mm, dy_mm, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_title(f"dy — 垂直位移\n(红=向下, 蓝=向上)\nmax |dy|={np.nanmax(np.abs(dy_mm[valid_mask])):.4f} mm", fontsize=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="mm", fraction=0.046)

    # |d|
    ax = axes[1, 0]
    im = ax.pcolormesh(X_mm, Y_mm, disp_mag, cmap="hot",
                        vmin=0, vmax=vmax, shading="auto")
    ax.set_title(f"|d| — 位移大小\n(越亮位移越大)\nmax={vmax:.4f} mm (95%分位)", fontsize=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    plt.colorbar(im, ax=ax, label="mm", fraction=0.046)

    # 箭头图
    ax = axes[1, 1]
    # 背景用 inpaint 后的图
    rect_contact_full = tac_utils.apply_warp(img_contact, map_x, map_y)
    rect_contact_inp_full = tac_utils.inpaint_rectified(rect_contact_full, meta)
    rect_contact_inp_crop, (crop_x0, crop_y0) = tac_utils.crop_rectified(rect_contact_inp_full, meta)
    ax.imshow(cv2.cvtColor(rect_contact_inp_crop, cv2.COLOR_BGR2RGB), alpha=0.4,
              extent=[0, tac_utils.N_COLS * tac_utils.PITCH_X_MM,
                      tac_utils.N_ROWS * tac_utils.PITCH_Y_MM, 0])
    q = ax.quiver(X_mm[valid_mask], Y_mm[valid_mask],
                  dx_mm[valid_mask], dy_mm[valid_mask],
                  disp_mag[valid_mask], cmap="hot", scale=vmax * 8, width=0.005)
    ax.set_title("位移向量场（quiver）\n箭头方向=位移方向，颜色深浅=大小", fontsize=12)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")
    plt.colorbar(q, ax=ax, label="|d| (mm)", fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "step05_heatmap.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 已保存: {save_path}")
    print()
    print("  看图提示：")
    print("    1. 先看 |d| 热力图，找到位移最大的区域")
    print("    2. 再看 dx/dy，分析是水平还是垂直方向为主")
    print("    3. 最后看 quiver 箭头图，直观感受位移方向")

    pause()

    # ============================================================
    # Step 6: 位移叠加图
    # ============================================================
    print_header("Step 6: 位移叠加在图像上（直观对照）")

    print("【做什么？】")
    print("  把位移值直接画在矫正后的图像上")
    print("  每个标记点用颜色表示位移大小/方向")
    print("  方便对照'这个位置的标记点位移了多少'")
    print()

    # 矫正+裁剪后的 contact 图像
    rect_crop, (cx0, cy0) = tac_utils.crop_rectified(tac_utils.apply_warp(img_contact, map_x, map_y), meta)
    rect_inp_crop, _ = tac_utils.crop_rectified(rect_contact_inp_full, meta)

    grid_contact_rect_crop = grid_contact_rect.copy()
    grid_contact_rect_crop[:, :, 0] -= cx0
    grid_contact_rect_crop[:, :, 1] -= cy0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.suptitle(f"位移叠加图\n{CONTACT_FILE}", fontsize=14, fontweight="bold")

    for i, (title, values, cmap) in enumerate([
        ("dx — 水平位移\n(蓝=向左, 红=向右)", dx_mm, "RdBu_r"),
        ("dy — 垂直位移\n(蓝=向上, 红=向下)", dy_mm, "RdBu_r"),
        ("|d| — 位移大小\n(越亮越大)", disp_mag, "hot"),
    ]):
        ax = axes[i]
        ax.imshow(cv2.cvtColor(rect_inp_crop, cv2.COLOR_BGR2RGB), alpha=0.7)
        vlim = vmax if "hot" not in cmap else vmax
        vmin = -vlim if "hot" not in cmap else 0
        sc = ax.scatter(grid_contact_rect_crop[valid_mask, 0],
                        grid_contact_rect_crop[valid_mask, 1],
                        c=values[valid_mask], cmap=cmap,
                        vmin=vmin, vmax=vlim,
                        s=35, alpha=0.85, edgecolors="white", linewidths=0.3)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, rect_inp_crop.shape[1])
        ax.set_ylim(rect_inp_crop.shape[0], 0)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        plt.colorbar(sc, ax=ax, label="mm", fraction=0.046)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "step06_overlay.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 已保存: {save_path}")

    pause()

    # ============================================================
    # Step 7: 总结
    # ============================================================
    print_header("Step 7: 总结回顾 — 你学会了什么？")

    print(f"""
【demo_09 核心知识点回顾】

1. 为什么要在矫正后坐标下算位移？
   → 矫正后 mm/px 均匀，位移物理意义明确

2. 正向点映射怎么做？
   → 三角剖分 + 仿射变换
   → 知道三角形三个顶点在两边的位置，就能算任意内点
   → 精度验证：ref 映射后与标准网格偏差 < 0.1 px

3. 位移怎么算？
   → displacement = contact_rect - standard_grid
   → 像素 → mm: 直接乘 mm_per_px

4. 三种可视化方式：
   → 热力图 (pcolormesh): 看整体分布
   → 箭头图 (quiver): 看方向
   → 叠加图 (scatter): 对照图像看细节

【关键数据汇总】
  网格大小: {tac_utils.N_ROWS}行 × {tac_utils.N_COLS}列 = {tac_utils.N_ROWS*tac_utils.N_COLS}个点
  有效点数: {n_valid}
  最大位移: {np.nanmax(disp_mag[valid_mask]):.4f} mm
  平均位移: {np.nanmean(disp_mag[valid_mask]):.4f} mm
  物理尺度: {rp['mm_per_px_x']:.5f} mm/px (x)

【下一步】
  demo_10: 批量验证 10 张图像，看看算法稳定性
""")

    print(f"  所有可视化图保存在: {OUTPUT_DIR}")
    print()
    print("✅ demo_09 逐步调试完成！")
    pause("按回车结束...")


if __name__ == "__main__":
    main()
