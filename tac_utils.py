"""
tac_utils.py — 视触觉传感器算法公共工具库
===========================================
所有检测、排序、几何矫正等核心逻辑均在此统一实现，
确保 Phase 1~Phase 5 使用完全一致的算法和参数。

核心函数：
  - detect_and_sort(img) → grid_points, dog_img, keypoints
  - build_warp(grid_points) → map_x, map_y, meta
  - apply_warp(img, map_x, map_y) → rectified

参数来源：Phase 1 几何矫正（demo_05_warp.py），经过验证。
"""

import cv2
import numpy as np
import json
import os

# ============================================================
# 全局配置参数（与 Phase 1 demo_05 完全一致）
# ============================================================
N_COLS = 16
N_ROWS = 21
N_EXPECTED = N_COLS * N_ROWS

ROI_X0, ROI_Y0 = 810, 225
ROI_X1, ROI_Y1 = 1140, 630

DOG_KERNEL_LARGE = 27
DOG_KERNEL_SMALL = 5
DOG_GAIN = 12.9

BLOB_PARAMS = dict(
    minThreshold=171.0,
    maxThreshold=255.0,
    thresholdStep=11.0,
    minDistBetweenBlobs=6.0,
    filterByArea=True,
    minArea=8.0,
    maxArea=1500.0,
    filterByCircularity=True,
    minCircularity=0.5,
    filterByConvexity=False,
    filterByInertia=False,
    blobColor=255,
)


# ============================================================
# 1. DoG + SimpleBlobDetector 检测标记点
# ============================================================
def detect_blobs(img):
    """
    从图像中检测标记点（DoG + SimpleBlobDetector）。
    参数与 Phase 1 demo_05 完全一致。

    参数:
        img: BGR 图像 (H, W, 3)

    返回:
        keypoints: cv2.KeyPoint 列表
        dog: DoG 图像（带 ROI 遮罩）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    blur_large = cv2.GaussianBlur(
        gray, (DOG_KERNEL_LARGE, DOG_KERNEL_LARGE),
        0.3 * ((DOG_KERNEL_LARGE - 1) * 0.5 - 1) + 0.8
    )
    blur_small = cv2.GaussianBlur(
        gray, (DOG_KERNEL_SMALL, DOG_KERNEL_SMALL),
        0.3 * ((DOG_KERNEL_SMALL - 1) * 0.5 - 1) + 0.8
    )
    dog = np.clip((blur_large - blur_small) * DOG_GAIN, 0.0, 255.0).astype(np.uint8)

    roi_mask = np.zeros_like(dog)
    roi_mask[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1] = 255
    dog = cv2.bitwise_and(dog, roi_mask)

    params = cv2.SimpleBlobDetector_Params()
    for k, v in BLOB_PARAMS.items():
        setattr(params, k, v)

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(dog)
    keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)[:N_EXPECTED]

    return keypoints, dog


# ============================================================
# 2. k-means 一维聚类
# ============================================================
def _kmeans_1d(values, k, iters=50):
    """一维 k-means（初始中心在 min~max 间均匀分布）"""
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


# ============================================================
# 3. PCA + k-means + 贪心分配 → 网格排序
# ============================================================
def sort_grid(keypoints):
    """
    将检测到的 keypoints 排序为规则网格。
    算法与 Phase 1 demo_05 完全一致。

    参数:
        keypoints: cv2.KeyPoint 列表

    返回:
        grid_points: (N_ROWS, N_COLS, 2) float64 数组，空单元格为 NaN
    """
    points = np.array(
        [(float(k.pt[0]), float(k.pt[1])) for k in keypoints],
        dtype=np.float64
    )

    # PCA
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    proj = centered @ R

    # 方向校正：主轴与图像坐标系一致
    if np.corrcoef(proj[:, 0], centered[:, 0])[0, 1] < 0:
        proj[:, 0] *= -1
    if np.corrcoef(proj[:, 1], centered[:, 1])[0, 1] < 0:
        proj[:, 1] *= -1

    # 判断哪个主成分对应列方向（水平）
    pc1_is_horizontal = abs(R[0, 0]) > abs(R[0, 1])
    if pc1_is_horizontal:
        col_base = proj[:, 0]
        row_base = proj[:, 1]
    else:
        col_base = proj[:, 1]
        row_base = proj[:, 0]

    # k-means 聚类
    x_centers = _kmeans_1d(col_base, N_COLS)
    y_centers = _kmeans_1d(row_base, N_ROWS)

    col_idx = np.argmin(np.abs(col_base[:, None] - x_centers[None, :]), axis=1)
    row_idx = np.argmin(np.abs(row_base[:, None] - y_centers[None, :]), axis=1)

    # 方向校正：列从左到右，行从上到下
    if np.corrcoef(col_idx, points[:, 0])[0, 1] < 0:
        col_idx = N_COLS - 1 - col_idx
        x_centers = x_centers[::-1]
    if np.corrcoef(row_idx, points[:, 1])[0, 1] < 0:
        row_idx = N_ROWS - 1 - row_idx
        y_centers = y_centers[::-1]

    # 贪心分配（按代价从小到大）
    costs = (col_base - x_centers[col_idx]) ** 2 + (row_base - y_centers[row_idx]) ** 2
    order_pts = np.argsort(costs)

    grid_points = np.full((N_ROWS, N_COLS, 2), np.nan, dtype=np.float64)
    for i in order_pts:
        r = int(row_idx[i])
        c = int(col_idx[i])
        if 0 <= r < N_ROWS and 0 <= c < N_COLS and np.isnan(grid_points[r, c, 0]):
            grid_points[r, c] = points[i]

    return grid_points


# ============================================================
# 4. 检测 + 排序 一键调用
# ============================================================
def detect_and_sort(img):
    """
    一键检测并排序标记点。

    参数:
        img: BGR 图像

    返回:
        grid_points: (N_ROWS, N_COLS, 2) float64
        dog: DoG 图像
        keypoints: 原始 keypoints 列表
    """
    keypoints, dog = detect_blobs(img)
    grid_points = sort_grid(keypoints)
    return grid_points, dog, keypoints


# ============================================================
# 5. 构建 Warp（完整 warp，与 demo_05 一致）
# ============================================================
def build_warp(grid_points, pitch_x_mm=0.6, pitch_y_mm=0.6,
               out_width=320, pad_px=20.0, mask_bottom_px=20):
    """
    从网格点构建稠密 warp 查找表。
    算法与 Phase 1 demo_05 完全一致。

    返回:
        map_x, map_y: (H, W) float32 查找表
        meta: 参数字典
        out_h: 输出图像高度
    """
    # 原始网格间距
    dx_raw_px = float(np.nanmedian(np.linalg.norm(
        grid_points[:, 1:, :] - grid_points[:, :-1, :], axis=2
    )))
    dy_raw_px = float(np.nanmedian(np.linalg.norm(
        grid_points[1:, :, :] - grid_points[:-1, :, :], axis=2
    )))

    if np.isnan(dx_raw_px):
        valid_x = grid_points[:, :, 0][~np.isnan(grid_points[:, :, 0])]
        dx_raw_px = (valid_x.max() - valid_x.min()) / (N_COLS - 1)
    if np.isnan(dy_raw_px):
        valid_y = grid_points[:, :, 1][~np.isnan(grid_points[:, :, 1])]
        dy_raw_px = (valid_y.max() - valid_y.min()) / (N_ROWS - 1)

    # 物理尺寸
    phys_w = (N_COLS - 1) * pitch_x_mm
    phys_h = (N_ROWS - 1) * pitch_y_mm
    aspect = phys_h / phys_w

    roi_rect_w = out_width - 2 * pad_px
    dx_rect = roi_rect_w / (N_COLS - 1)
    dy_rect = dx_rect * aspect

    out_h_float = 2 * pad_px + (N_ROWS - 1) * dy_rect
    out_h = int(np.ceil(out_h_float))

    scale_x = dx_rect / dx_raw_px
    scale_y = dy_rect / dy_raw_px

    # 边界外推
    def extend_grid_boundary(gp, pad_x, pad_y):
        rows, cols, _ = gp.shape
        row_ext = []
        for r in range(rows):
            row = gp[r]
            left_dir = row[1] - row[0]
            left_norm = np.linalg.norm(left_dir)
            if left_norm > 1e-9:
                left = row[0] - left_dir / left_norm * pad_x
            else:
                left = row[0] - np.array([pad_x, 0])
            right_dir = row[-1] - row[-2]
            right_norm = np.linalg.norm(right_dir)
            if right_norm > 1e-9:
                right = row[-1] + right_dir / right_norm * pad_x
            else:
                right = row[-1] + np.array([pad_x, 0])
            row_ext.append(np.vstack([left, row, right]))
        gp_lr = np.stack(row_ext, axis=0)

        col_ext = []
        for c in range(cols + 2):
            col = gp_lr[:, c]
            top_dir = col[1] - col[0]
            top_norm = np.linalg.norm(top_dir)
            if top_norm > 1e-9:
                top = col[0] - top_dir / top_norm * pad_y
            else:
                top = col[0] - np.array([0, pad_y])
            bot_dir = col[-1] - col[-2]
            bot_norm = np.linalg.norm(bot_dir)
            if bot_norm > 1e-9:
                bot = col[-1] + bot_dir / bot_norm * pad_y
            else:
                bot = col[-1] + np.array([0, pad_y])
            col_ext.append(np.vstack([top, col, bot]))
        return np.stack(col_ext, axis=1)

    raw_ext = extend_grid_boundary(grid_points, dx_raw_px * 1.5, dy_raw_px * 1.5)

    # 标准网格
    xs = np.concatenate([
        [0.0],
        pad_px + np.arange(N_COLS, dtype=np.float64) * dx_rect,
        [float(out_width)]
    ])
    ys = np.concatenate([
        [0.0],
        pad_px + np.arange(N_ROWS, dtype=np.float64) * dy_rect,
        [float(out_h)]
    ])
    can_pts = np.zeros((N_ROWS + 2, N_COLS + 2, 2), dtype=np.float64)
    for r in range(N_ROWS + 2):
        for c in range(N_COLS + 2):
            can_pts[r, c] = (xs[c], ys[r])

    # 逐三角形仿射映射
    def point_in_triangle(px, py, tri):
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

    def dense_map_from_grids(raw_pts, can_pts, h, w):
        raw_pts = np.asarray(raw_pts, dtype=np.float32)
        can_pts = np.asarray(can_pts, dtype=np.float32)
        map_x = np.full((h, w), -1.0, dtype=np.float32)
        map_y = np.full((h, w), -1.0, dtype=np.float32)
        rows, cols, _ = can_pts.shape
        for r in range(rows - 1):
            for c in range(cols - 1):
                q00 = can_pts[r, c]
                q10 = can_pts[r, c + 1]
                q01 = can_pts[r + 1, c]
                q11 = can_pts[r + 1, c + 1]
                p00 = raw_pts[r, c]
                p10 = raw_pts[r, c + 1]
                p01 = raw_pts[r + 1, c]
                p11 = raw_pts[r + 1, c + 1]
                tri_q = [(q00, q10, q11), (q00, q11, q01)]
                tri_p = [(p00, p10, p11), (p00, p11, p01)]
                for tq, tp in zip(tri_q, tri_p):
                    tq = np.asarray(tq, dtype=np.float32)
                    tp = np.asarray(tp, dtype=np.float32)
                    A = cv2.getAffineTransform(tq, tp)
                    x_min = int(max(0, np.floor(np.min(tq[:, 0]))))
                    x_max = int(min(w - 1, np.ceil(np.max(tq[:, 0]))))
                    y_min = int(max(0, np.floor(np.min(tq[:, 1]))))
                    y_max = int(min(h - 1, np.ceil(np.max(tq[:, 1]))))
                    if x_max < x_min or y_max < y_min:
                        continue
                    xs_arr = np.arange(x_min, x_max + 1)
                    ys_arr = np.arange(y_min, y_max + 1)
                    xv, yv = np.meshgrid(xs_arr, ys_arr)
                    inside = point_in_triangle(
                        xv.astype(np.float32), yv.astype(np.float32), tq
                    )
                    if not np.any(inside):
                        continue
                    x_flat = xv[inside].astype(np.float32)
                    y_flat = yv[inside].astype(np.float32)
                    u = A[0, 0] * x_flat + A[0, 1] * y_flat + A[0, 2]
                    v = A[1, 0] * x_flat + A[1, 1] * y_flat + A[1, 2]
                    map_x[y_flat.astype(np.int32), x_flat.astype(np.int32)] = u
                    map_y[y_flat.astype(np.int32), x_flat.astype(np.int32)] = v
        return map_x, map_y

    map_x, map_y = dense_map_from_grids(raw_ext, can_pts, out_h, out_width)

    # 保存三角形信息用于正向点映射（raw → rect）
    # 每个三角形：can 中的三角形和对应的 raw 中三角形，以及逆仿射变换
    tris = []  # list of (tri_can_3x2, tri_raw_3x2, A_inv_2x3)
    rows, cols = can_pts.shape[0], can_pts.shape[1]
    for r in range(rows - 1):
        for c in range(cols - 1):
            q00 = can_pts[r, c]
            q10 = can_pts[r, c + 1]
            q01 = can_pts[r + 1, c]
            q11 = can_pts[r + 1, c + 1]
            p00 = raw_ext[r, c]
            p10 = raw_ext[r, c + 1]
            p01 = raw_ext[r + 1, c]
            p11 = raw_ext[r + 1, c + 1]
            tri_pairs = [
                (np.array([q00, q10, q11], dtype=np.float32),
                 np.array([p00, p10, p11], dtype=np.float32)),
                (np.array([q00, q11, q01], dtype=np.float32),
                 np.array([p00, p11, p01], dtype=np.float32)),
            ]
            for tq, tp in tri_pairs:
                # A: can → raw (2x3)
                A = cv2.getAffineTransform(tq, tp)
                # A_inv: raw → can (2x3)
                A_full = np.eye(3, dtype=np.float64)
                A_full[:2, :] = A
                try:
                    A_inv_full = np.linalg.inv(A_full)
                except np.linalg.LinAlgError:
                    continue
                A_inv = A_inv_full[:2, :].astype(np.float32)
                tris.append({
                    "tri_raw": tp,
                    "tri_can": tq,
                    "A_inv": A_inv,
                })

    meta = {
        "grid": {"n_cols": N_COLS, "n_rows": N_ROWS,
                 "pitch_x_mm": pitch_x_mm, "pitch_y_mm": pitch_y_mm},
        "rectified_px": {
            "out_width": out_width, "out_height": out_h,
            "pad_px": pad_px,
            "dx_raw_px": dx_raw_px, "dy_raw_px": dy_raw_px,
            "dx_rect": dx_rect, "dy_rect": dy_rect,
            "scale_x": scale_x, "scale_y": scale_y,
            "mm_per_px_x": pitch_x_mm / (dx_raw_px * scale_x),
            "mm_per_px_y": pitch_y_mm / (dy_raw_px * scale_y),
        },
        "tris": tris,
    }

    return map_x, map_y, meta, out_h


# ============================================================
# 6. 应用 Warp
# ============================================================
def apply_warp(img, map_x, map_y):
    """
    用 warp 查找表矫正图像。

    返回:
        rectified: 矫正后的 BGR 图像
    """
    return cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


# ============================================================
# 7. 保存/加载 Warp
# ============================================================
def save_warp(path, map_x, map_y, meta):
    np.savez_compressed(path, map_x=map_x, map_y=map_y, meta=meta)


def load_warp(path):
    data = np.load(path, allow_pickle=True)
    map_x = data['map_x']
    map_y = data['map_y']
    meta = data['meta'].item()
    return map_x, map_y, meta


# ============================================================
# 8. 矫正后图像裁剪
# ============================================================
WARP_OUT_WIDTH = 320
WARP_PAD_PX = 20.0
WARP_MASK_BOTTOM_PX = 20
PITCH_X_MM = 0.6
PITCH_Y_MM = 0.6


def crop_rectified(rectified, meta, mask_bottom_px=WARP_MASK_BOTTOM_PX):
    """
    裁剪矫正后图像到标记区域（去掉黑边和底部遮挡）。

    参数:
        rectified: apply_warp 输出的矫正图像
        meta: build_warp 返回的 meta 字典
        mask_bottom_px: 底部裁剪像素数

    返回:
        cropped: 裁剪后的 BGR 图像
        (x0, y0): 裁剪左上角在 rectified 中的偏移
    """
    rp = meta["rectified_px"]
    pad_px = rp["pad_px"]
    dx_rect = rp["dx_rect"]
    dy_rect = rp["dy_rect"]

    u0 = pad_px
    v0 = pad_px
    u1 = pad_px + (N_COLS - 1) * dx_rect
    v1 = pad_px + (N_ROWS - 1) * dy_rect

    x0 = max(0, int(np.floor(min(u0, u1) - pad_px)))
    x1 = min(rectified.shape[1], int(np.ceil(max(u0, u1) + pad_px)))
    y0 = max(0, int(np.floor(min(v0, v1) - pad_px)))
    y1 = min(rectified.shape[0], int(np.ceil(max(v0, v1) + pad_px)) - int(mask_bottom_px))

    return rectified[y0:y1, x0:x1], (x0, y0)


# ============================================================
# 9. 矫正后标准网格坐标
# ============================================================
def get_rectified_grid_centers(meta):
    """
    获取矫正后图像中标记点的标准位置（均匀网格）。

    返回:
        centers: (N_ROWS, N_COLS, 2) float64，(u, v) 像素坐标
        dx, dy: 矫正后列/行间距（像素）
    """
    rp = meta["rectified_px"]
    pad_px = rp["pad_px"]
    dx = rp["dx_rect"]
    dy = rp["dy_rect"]

    centers = np.zeros((N_ROWS, N_COLS, 2), dtype=np.float64)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            centers[r, c] = (pad_px + c * dx, pad_px + r * dy)
    return centers, dx, dy


# ============================================================
# 10. 标记点遮罩生成（原始图像坐标）
# ============================================================
def draw_marker_mask_raw(grid_points, radius=4.0):
    """
    在原始图像尺寸上，根据检测到的网格点位置绘制圆形遮罩。

    参数:
        grid_points: (N_ROWS, N_COLS, 2) 检测到的网格点（NaN为空）
        radius: 遮罩圆半径（像素）

    返回:
        mask: (1080, 1920) uint8 遮罩，标记点处为255
    """
    h, w = 1080, 1920
    mask = np.zeros((h, w), dtype=np.uint8)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            pt = grid_points[r, c]
            if not np.isnan(pt[0]):
                cv2.circle(mask, (int(round(pt[0])), int(round(pt[1]))),
                           int(round(radius)), 255, -1)
    return mask


# ============================================================
# 11. 标记点遮罩生成（矫正后图像坐标）
# ============================================================
def draw_marker_mask_rectified(meta, radius_scale=1.2):
    """
    在矫正后图像上，根据标准网格位置绘制圆形遮罩。

    参数:
        meta: build_warp 返回的 meta 字典
        radius_scale: 半径相对于网格间距的比例

    返回:
        mask: (out_h, out_w) uint8 遮罩
    """
    rp = meta["rectified_px"]
    out_h = rp["out_height"]
    out_w = rp["out_width"]
    dx = rp["dx_rect"]
    dy = rp["dy_rect"]
    radius = int(round(min(dx, dy) * radius_scale * 0.5))

    centers, _, _ = get_rectified_grid_centers(meta)
    mask = np.zeros((out_h, out_w), dtype=np.uint8)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            u, v = centers[r, c]
            cv2.circle(mask, (int(round(u)), int(round(v))), radius, 255, -1)
    return mask


# ============================================================
# 12. Inpaint 标记修复
# ============================================================
def inpaint_bgr(img, mask, radius=3):
    """
    使用 Telea 算法修复标记点区域。

    参数:
        img: BGR 图像
        mask: uint8 遮罩（255=需要修复的区域）
        radius: inpaint 半径

    返回:
        修复后的 BGR 图像
    """
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    return cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)


def inpaint_raw(img, grid_points, marker_radius=4.0, inpaint_radius=3):
    """
    在原始图像上检测标记点并修复。

    参数:
        img: BGR 原始图像
        grid_points: (N_ROWS, N_COLS, 2) 检测到的网格点
        marker_radius: 标记点遮罩半径
        inpaint_radius: inpaint 算法半径

    返回:
        修复后的 BGR 图像
    """
    mask = draw_marker_mask_raw(grid_points, radius=marker_radius)
    return inpaint_bgr(img, mask, radius=inpaint_radius)


def inpaint_rectified(rectified, meta, inpaint_radius=3):
    """
    在矫正后图像上修复标记点（使用标准网格位置）。

    参数:
        rectified: 矫正后的 BGR 图像
        meta: build_warp 返回的 meta
        inpaint_radius: inpaint 算法半径

    返回:
        修复后的 BGR 图像
    """
    mask = draw_marker_mask_rectified(meta)
    return inpaint_bgr(rectified, mask, radius=inpaint_radius)


# ============================================================
# 13. 位移场计算
# ============================================================
def compute_displacement(grid_ref, grid_contact):
    """
    计算从参考网格到接触网格的位移场。

    参数:
        grid_ref: (N_ROWS, N_COLS, 2) 参考网格点
        grid_contact: (N_ROWS, N_COLS, 2) 接触网格点

    返回:
        disp: (N_ROWS, N_COLS, 2) 位移向量 (dx, dy)，无效点为 NaN
        valid_mask: (N_ROWS, N_COLS) bool，True 表示该点有效
    """
    disp = grid_contact - grid_ref
    valid_mask = ~(np.isnan(disp[:, :, 0]) | np.isnan(disp[:, :, 1]))
    disp[~valid_mask] = np.nan
    return disp, valid_mask


def disp_to_physical(disp, meta):
    """
    将像素位移转换为物理位移（mm）。

    参数:
        disp: (N_ROWS, N_COLS, 2) 像素位移
        meta: build_warp 返回的 meta

    返回:
        disp_mm: (N_ROWS, N_COLS, 2) 物理位移 (dx_mm, dy_mm)
    """
    mmpx_x = meta["rectified_px"]["mm_per_px_x"]
    mmpx_y = meta["rectified_px"]["mm_per_px_y"]
    disp_mm = np.empty_like(disp)
    disp_mm[:, :, 0] = disp[:, :, 0] * mmpx_x
    disp_mm[:, :, 1] = disp[:, :, 1] * mmpx_y
    return disp_mm


# ============================================================
# 14. 正向点映射：原始坐标 → 矫正后坐标（利用三角剖分逆变换）
# ============================================================
def warp_points_raw_to_rect(points_raw, meta):
    """
    将原始图像坐标中的点映射到矫正后坐标。
    使用 build_warp 时保存的三角剖分及逆仿射变换。

    参数:
        points_raw: (..., 2) 原始图像坐标 (x, y)，可以包含 NaN
        meta: build_warp 返回的 meta（包含 tris）

    返回:
        points_rect: (..., 2) 矫正后坐标 (u, v)，未找到映射的点为 NaN
    """
    tris = meta["tris"]
    shape = points_raw.shape
    pts_flat = points_raw.reshape(-1, 2).astype(np.float64)
    out_flat = np.full_like(pts_flat, np.nan)

    for i, pt in enumerate(pts_flat):
        if np.isnan(pt[0]) or np.isnan(pt[1]):
            continue
        x, y = pt[0], pt[1]
        for tri in tris:
            tp = tri["tri_raw"]  # (3,2) 三角形在原始图像中的坐标
            # 检查点是否在三角形内（barycentric）
            x0, y0 = tp[0]
            x1, y1 = tp[1]
            x2, y2 = tp[2]
            v0x, v0y = x2 - x0, y2 - y0
            v1x, v1y = x1 - x0, y1 - y0
            v2x, v2y = x - x0, y - y0
            den = v0x * v1y - v1x * v0y
            if abs(den) < 1e-9:
                continue
            a = (v2x * v1y - v1x * v2y) / den
            b = (v0x * v2y - v2x * v0y) / den
            c = 1.0 - a - b
            if a >= -1e-4 and b >= -1e-4 and c >= -1e-4:
                A_inv = tri["A_inv"]
                u = A_inv[0, 0] * x + A_inv[0, 1] * y + A_inv[0, 2]
                v = A_inv[1, 0] * x + A_inv[1, 1] * y + A_inv[1, 2]
                out_flat[i] = (u, v)
                break

    return out_flat.reshape(shape)
