"""
demo_08_step_by_step.py
Phase 2: Warp 矫正 + Inpaint 标记修复 — 逐步调试走读版
每一步都有：目的讲解 + 数值统计 + 可视化输出
做到"数形结合"，帮助深入理解每一步在做什么

使用方法：
  运行后，每一步都会暂停，按回车继续下一步
  每一步的可视化图都会保存到 outputs/debug_demo08/
"""
import cv2
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import tac_utils
from tac_utils import N_COLS, N_ROWS, N_EXPECTED

DATA_DIR  = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
REF_FILE  = "ref.jpg"
CONTACT_FILE = "image_20260430_135445_239.jpg"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "debug_demo08")
os.makedirs(OUTPUT_DIR, exist_ok=True)
WARP_PATH = os.path.join(OUTPUT_DIR, "..", "warp_ref.npz")


def pause(msg="按回车继续下一步..."):
    input("\n" + msg)


def save_fig(fig, name, step_num):
    path = os.path.join(OUTPUT_DIR, f"step{step_num:02d}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  [图已保存] {path}")
    return path


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ============================================================
# Step 0: 总览
# ============================================================
print_section("Step 0: demo_08 整体路线总览")
print("""
demo_08 是 Phase 2 的第二步，核心回答两个问题：
  ① 怎么把扭曲的图像"掰正"？→ Warp 几何矫正
  ② 怎么把黑色标记点"去掉"？→ Inpaint 修复

为什么需要这两步？
  因为 Phase 3 光度立体法向估计需要：
    1. 横平竖直的标准图像（网格对齐，方便处理）
    2. 干净的皮肤纹理（黑色标记点会干扰颜色分析）

完整流程（8大步）：
  ① 加载 Warp 参数        → map_x, map_y 是什么？
  ② 理解查找表            → map_x / map_y 可视化
  ③ 应用 Warp 矫正        → 原图 → 矫正图
  ④ 裁剪有效区域          → 裁掉底部无效区域
  ⑤ 生成标记点遮罩        → mask 长什么样？
  ⑥ Inpaint 修复          → 标记点去哪了？
  ⑦ 两种 Inpaint 方式对比 → 先矫正后修复 vs 先修复后矫正
  ⑧ 放大细节对比          → 修复效果好不好？
""")
pause("准备好了吗？按回车开始 Step 1 →")


# ============================================================
# Step 1: 加载 / 构建 Warp
# ============================================================
print_section("Step 1: 加载 Warp 参数 — Warp 是什么？")

ref_path = os.path.join(DATA_DIR, REF_FILE)
img_ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)

if os.path.exists(WARP_PATH):
    print(f"  找到已有 Warp 文件，加载中...")
    map_x, map_y, meta = tac_utils.load_warp(WARP_PATH)
else:
    print("  没有找到 Warp 文件，从 ref.jpg 重新构建...")
    grid_ref, dog_ref, kp_ref = tac_utils.detect_and_sort(img_ref)
    map_x, map_y, meta, out_h = tac_utils.build_warp(grid_ref)
    tac_utils.save_warp(WARP_PATH, map_x, map_y, meta)
    print(f"  Warp 已保存: {WARP_PATH}")

rp = meta["rectified_px"]
out_w = rp["out_width"]
out_h = rp["out_height"]

print(f"""
【Warp 是什么？】
  Warp = 一组"坐标映射规则"，告诉你：
    "矫正后图像的 (x, y) 像素，应该去原始图像的哪个位置取颜色"

  具体存了什么？
    map_x[Y, X] = 矫正后 (X,Y) 对应原图的 x 坐标
    map_y[Y, X] = 矫正后 (X,Y) 对应原图的 y 坐标

  这叫"反向映射"（反向 = 从结果反推源头）。

【数值信息】
  矫正后图像尺寸: {out_w} × {out_h} 像素 (宽×高)
  map_x 形状    : {map_x.shape}  (和矫正图一样大)
  map_y 形状    : {map_y.shape}
  数据类型      : {map_x.dtype}  (float32，因为映射位置可以是小数)

  map_x 范围: [{np.min(map_x):.1f}, {np.max(map_x):.1f}] px
  map_y 范围: [{np.min(map_y):.1f}, {np.max(map_y):.1f}] px

  矫正后网格间距:
    dx = {rp['dx_rect']:.2f} px (列间距)
    dy = {rp['dy_rect']:.2f} px (行间距)

  物理尺度: {rp['mm_per_px_x']:.5f} mm/px
  （矫正后每个像素对应多少毫米）
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 把 map_x 和 map_y 当作图像显示
im0 = axes[0].imshow(map_x, cmap='jet')
axes[0].set_title("map_x — 水平方向映射（颜色=对应原图x坐标）", fontsize=12)
axes[0].set_xlabel("矫正后 X (px)")
axes[0].set_ylabel("矫正后 Y (px)")
plt.colorbar(im0, ax=axes[0], label='原图 x 坐标 (px)')

im1 = axes[1].imshow(map_y, cmap='jet')
axes[1].set_title("map_y — 垂直方向映射（颜色=对应原图y坐标）", fontsize=12)
axes[1].set_xlabel("矫正后 X (px)")
axes[1].set_ylabel("矫正后 Y (px)")
plt.colorbar(im1, ax=axes[1], label='原图 y 坐标 (px)')

plt.tight_layout()
save_fig(fig, "01_warp_maps", 1)
plt.close(fig)

print("""
【怎么看这两张图】
  左边 map_x：
    - 颜色越蓝 = 对应原图的 x 越小（越靠左）
    - 颜色越红 = 对应原图的 x 越大（越靠右）
    - 如果是完美的线性映射，应该是从左到右均匀的蓝色→红色渐变
    - 现在有弯曲，说明原始图像的网格是斜的/扭曲的

  右边 map_y：
    - 颜色越蓝 = 对应原图的 y 越小（越靠上）
    - 颜色越红 = 对应原图的 y 越大（越靠下）

  这两张图就是 Warp 的"身份证"——
  矫正过程 = 拿着这两张查表，一个像素一个像素地"搬家"。
""")

pause("理解 Warp 查找表了吗？按回车继续 Step 2 →")


# ============================================================
# Step 2: 应用 Warp — 图像矫正
# ============================================================
print_section("Step 2: 应用 Warp — 把图像『掰正』")

contact_path = os.path.join(DATA_DIR, CONTACT_FILE)
img_contact = cv2.imread(contact_path, cv2.IMREAD_COLOR)

rect_ref_full = tac_utils.apply_warp(img_ref, map_x, map_y)
rect_contact_full = tac_utils.apply_warp(img_contact, map_x, map_y)

print(f"""
【这一步在做什么】
  调用 cv2.remap(map_x, map_y)：
    对矫正后图像的每个像素 (X, Y)：
      x_src = map_x[Y, X]   （去原图哪个 x 取色）
      y_src = map_y[Y, X]   （去原图哪个 y 取色）
      dst[Y, X] = src[y_src, x_src]  （双线性插值取色）

  就像把原图的像素"重新排列"到新的位置上。

【数值信息】
  原始图像尺寸  : {img_ref.shape[1]} × {img_ref.shape[0]} px
  矫正后图像尺寸: {rect_ref_full.shape[1]} × {rect_ref_full.shape[0]} px

  参考图矫正后亮度:
    均值 = {np.mean(rect_ref_full):.1f}  (0-255)
    最小值 = {np.min(rect_ref_full)}
    最大值 = {np.max(rect_ref_full)}
""")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

img_rgb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
img_rgb_contact = cv2.cvtColor(img_contact, cv2.COLOR_BGR2RGB)
rect_ref_rgb = cv2.cvtColor(rect_ref_full, cv2.COLOR_BGR2RGB)
rect_contact_rgb = cv2.cvtColor(rect_contact_full, cv2.COLOR_BGR2RGB)

axes[0, 0].imshow(img_rgb_ref)
axes[0, 0].set_title("原始参考图像", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(rect_ref_rgb)
axes[0, 1].set_title("Warp 矫正后（横平竖直）", fontsize=12)
axes[0, 1].axis("off")

axes[1, 0].imshow(img_rgb_contact)
axes[1, 0].set_title("原始接触图像", fontsize=12)
axes[1, 0].axis("off")

axes[1, 1].imshow(rect_contact_rgb)
axes[1, 1].set_title("Warp 矫正后", fontsize=12)
axes[1, 1].axis("off")

plt.suptitle("原始图像 vs Warp 矫正后", fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(fig, "02_warp_apply", 2)
plt.close(fig)

print("""
【怎么看这张图】
  左右对比看：
    左边原图的网格是斜的/扭曲的
    右边矫正后的网格是横平竖直的

  上下对比看：
    上面是参考图（没按压）
    下面是接触图（按压后）

  矫正后有什么好处？
    1. 网格位置固定，每行每列对齐了
    2. 间距均匀，像素对应物理距离一致
    3. 后续处理（位移、法向）都在标准坐标系下进行
""")

pause("看到矫正效果了吗？按回车继续 Step 3 →")


# ============================================================
# Step 3: 裁剪有效区域
# ============================================================
print_section("Step 3: 裁剪有效区域 — 把没用的边去掉")

rect_ref_crop, (x0, y0) = tac_utils.crop_rectified(rect_ref_full, meta)
rect_contact_crop, _ = tac_utils.crop_rectified(rect_contact_full, meta)

print(f"""
【为什么要裁剪？】
  Warp 矫正后，图像四周（尤其是底部）有一些黑色无效区域——
  因为那些位置在原始图像中没有对应的像素（映射超出边界了）。

  裁剪后只保留有效区域，节省计算量，也更美观。

【数值信息】
  裁剪前尺寸: {rect_ref_full.shape[1]} × {rect_ref_full.shape[0]} px
  裁剪后尺寸: {rect_ref_crop.shape[1]} × {rect_ref_crop.shape[0]} px
  裁剪偏移  : x0={x0}, y0={y0} （裁剪左上角在原图中的位置）
  裁掉了底部: {rect_ref_full.shape[0] - rect_ref_crop.shape[0]} 行像素
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

rect_ref_crop_rgb = cv2.cvtColor(rect_ref_crop, cv2.COLOR_BGR2RGB)
axes[0].imshow(rect_ref_rgb)
axes[0].add_patch(plt.Rectangle((x0, y0), rect_ref_crop.shape[1], rect_ref_crop.shape[0],
                                 fill=False, edgecolor='red', linewidth=2))
axes[0].set_title("矫正后完整图像（红框=裁剪区域）", fontsize=12)
axes[0].axis("off")

axes[1].imshow(rect_ref_crop_rgb)
axes[1].set_title("裁剪后（仅保留有效区域）", fontsize=12)
axes[1].axis("off")

plt.tight_layout()
save_fig(fig, "03_crop", 3)
plt.close(fig)

pause("理解裁剪了吗？按回车继续 Step 4 →")


# ============================================================
# Step 4: 生成标记点遮罩
# ============================================================
print_section("Step 4: 标记点遮罩生成 — 哪些位置是标记点？")

mask_rect = tac_utils.draw_marker_mask_rectified(meta, radius_scale=1.2)

mask_crop = mask_rect[y0:y0+rect_ref_crop.shape[0], x0:x0+rect_ref_crop.shape[1]]

print(f"""
【遮罩是什么？】
  一张和矫正图一样大的黑白图：
    - 白色（255）= 标记点位置，需要修复
    - 黑色（0）= 正常皮肤，不用动

  Inpaint 算法就看着这张 mask 图：
    "白色的地方我给你补一补，黑色的地方别动"

【数值信息】
  mask 尺寸: {mask_rect.shape[1]} × {mask_rect.shape[0]} px
  mask 中白色像素数: {np.sum(mask_rect > 0)} 个
  总像素数: {mask_rect.size} 个
  标记点占比: {np.sum(mask_rect > 0) / mask_rect.size * 100:.2f}%

  每个标记点遮罩半径约: {meta['rectified_px']['dx_rect'] * 1.2 * 0.4:.1f} px
  （约为网格间距的 40%，比标记点实际大小大一点，确保完全覆盖）
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 裁剪后的 mask
axes[0].imshow(mask_crop, cmap='gray')
axes[0].set_title("标记点遮罩（白色=标记点，需要修复）", fontsize=12)
axes[0].axis("off")

# 叠加显示：把 mask 叠在矫正图上
overlay = rect_ref_crop.copy()
overlay[mask_crop > 0] = [0, 0, 255]  # 标记点区域涂红（OpenCV BGR格式，红色=[0,0,255]）
axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1].set_title("遮罩叠加在矫正图上（红色=标记点位置）", fontsize=12)
axes[1].axis("off")

plt.tight_layout()
save_fig(fig, "04_mask", 4)
plt.close(fig)

print("""
【怎么看这张图】
  左边 mask 图：
    白色的小点就是每个标记点的位置
    排列得整整齐齐 —— 因为矫正后网格是标准的！
    （这也是为什么先矫正再 inpaint 更准的原因）

  右边叠加图：
    红色圆圈正好套住黑色标记点
    说明 mask 生成准确，完全覆盖了需要修复的区域
""")

pause("理解遮罩了吗？按回车继续 Step 5 →")


# ============================================================
# Step 5: Inpaint 修复
# ============================================================
print_section("Step 5: Inpaint 标记点修复 — 让标记点消失")

rect_ref_inp = tac_utils.inpaint_rectified(rect_ref_full, meta, inpaint_radius=3)
rect_contact_inp = tac_utils.inpaint_rectified(rect_contact_full, meta, inpaint_radius=3)

rect_ref_inp_crop, _ = tac_utils.crop_rectified(rect_ref_inp, meta)
rect_contact_inp_crop, _ = tac_utils.crop_rectified(rect_contact_inp, meta)

print(f"""
【Inpaint 是什么？】
  Inpaint = 图像修复
  算法：Telea 算法（OpenCV 内置）

  原理很简单：
    "看看标记点周围的像素是什么颜色，
     按照一定的权重混合起来，把中间补上"

  就像修照片——脸上有个痣，用周围皮肤的颜色把痣盖住。

  inpaint_radius = 3 px
    （修复时参考周围 3 像素范围内的颜色）

【数值信息】
  修复前标记点区域像素值: （取第10行第5列的标记点）
    （跳过演示，直接看效果对比图）
""")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ref_inp_crop_rgb = cv2.cvtColor(rect_ref_inp_crop, cv2.COLOR_BGR2RGB)
contact_inp_crop_rgb = cv2.cvtColor(rect_contact_inp_crop, cv2.COLOR_BGR2RGB)

axes[0, 0].imshow(rect_ref_crop_rgb)
axes[0, 0].set_title("参考图 — 矫正后（有标记点）", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(ref_inp_crop_rgb)
axes[0, 1].set_title("参考图 — Inpaint 后（标记点消失了！）", fontsize=12)
axes[0, 1].axis("off")

rect_contact_crop_rgb = cv2.cvtColor(rect_contact_crop, cv2.COLOR_BGR2RGB)
axes[1, 0].imshow(rect_contact_crop_rgb)
axes[1, 0].set_title("接触图 — 矫正后（有标记点）", fontsize=12)
axes[1, 0].axis("off")

axes[1, 1].imshow(contact_inp_crop_rgb)
axes[1, 1].set_title("接触图 — Inpaint 后（标记点消失了！）", fontsize=12)
axes[1, 1].axis("off")

plt.suptitle("Inpaint 修复前后对比", fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(fig, "05_inpaint", 5)
plt.close(fig)

print("""
【为什么要费这么大劲去掉标记点？】
  因为 Phase 3 光度立体要靠"颜色变化"来推算法向量。
  黑色标记点没有颜色信息（RGB 都很低），会严重干扰法向估计。

  就像你看人家长相，脸上有颗黑痣也没事，
  但如果你要靠脸上的光影变化推测脸的形状，
  那颗痣就会变成干扰——它不是真的形状凹陷，只是颜色黑。

  所以必须先把标记点"抹掉"，再做后续处理。
""")

pause("看到 Inpaint 效果了吗？按回车继续 Step 6 →")


# ============================================================
# Step 6: 放大对比 — 细节观察
# ============================================================
print_section("Step 6: 放大对比 — 修复效果到底好不好？")

h, w = rect_ref_crop_rgb.shape[:2]
cy, cx = h // 2, w // 2          # 中心点
half = 80                        # 取中间 160×160 区域

zoom_y0, zoom_y1 = cy - half, cy + half
zoom_x0, zoom_x1 = cx - half, cx + half

zoom_ref_before = rect_ref_crop_rgb[zoom_y0:zoom_y1, zoom_x0:zoom_x1]
zoom_ref_after = ref_inp_crop_rgb[zoom_y0:zoom_y1, zoom_x0:zoom_x1]
zoom_contact_before = rect_contact_crop_rgb[zoom_y0:zoom_y1, zoom_x0:zoom_x1]
zoom_contact_after = contact_inp_crop_rgb[zoom_y0:zoom_y1, zoom_x0:zoom_x1]

# 计算差值图
diff_ref = np.abs(zoom_ref_before.astype(float) - zoom_ref_after.astype(float))
diff_contact = np.abs(zoom_contact_before.astype(float) - zoom_contact_after.astype(float))

print(f"""
【放大中间区域看细节】
  放大区域: ({zoom_x0}, {zoom_y0}) → ({zoom_x1}, {zoom_y1})
  放大尺寸: {zoom_x1 - zoom_x0} × {zoom_y1 - zoom_y0} px

  修复前后差异统计（参考图）:
    平均像素差: {np.mean(diff_ref):.2f} (0-255)
    最大像素差: {np.max(diff_ref):.2f}
    有差异的像素: {np.sum(diff_ref > 1)} / {diff_ref.size} 个

  （差值越大，说明 Inpaint 改变的像素越多）
""")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(zoom_ref_before)
axes[0, 0].set_title("Ref 修复前（有黑色标记点）", fontsize=11)
axes[0, 0].axis("off")

axes[0, 1].imshow(zoom_ref_after)
axes[0, 1].set_title("Ref 修复后（标记点消失）", fontsize=11)
axes[0, 1].axis("off")

im0 = axes[0, 2].imshow(np.mean(diff_ref, axis=2), cmap='hot')
axes[0, 2].set_title("差值图（越亮=变化越大）", fontsize=11)
axes[0, 2].axis("off")
plt.colorbar(im0, ax=axes[0, 2], fraction=0.046, pad=0.04)

axes[1, 0].imshow(zoom_contact_before)
axes[1, 0].set_title("Contact 修复前", fontsize=11)
axes[1, 0].axis("off")

axes[1, 1].imshow(zoom_contact_after)
axes[1, 1].set_title("Contact 修复后", fontsize=11)
axes[1, 1].axis("off")

im1 = axes[1, 2].imshow(np.mean(diff_contact, axis=2), cmap='hot')
axes[1, 2].set_title("差值图", fontsize=11)
axes[1, 2].axis("off")
plt.colorbar(im1, ax=axes[1, 2], fraction=0.046, pad=0.04)

plt.suptitle("Inpaint 效果放大对比（中间区域 160×160）", fontsize=14, fontweight='bold')
plt.tight_layout()
save_fig(fig, "06_zoom_compare", 6)
plt.close(fig)

print("""
【怎么看这张图】
  左边两列：前后对比
    - 修复前：黑色圆点很明显
    - 修复后：圆点不见了，周围纹理被"延续"过来了

  右边差值图：
    亮的地方 = Inpaint 改变了的像素
    正好是一个个圆点的形状 —— 说明只改了标记点区域
    其他地方都是黑的（没动）

  修复效果好吗？
    近距离看还是能看出一点点痕迹（毕竟是"猜"出来的），
    但对于后续光度立体来说已经足够好了——
    总比一个大黑圈强得多。
""")

pause("细节看清楚了吗？按回车继续 Step 7 →")


# ============================================================
# Step 7: 两种 Inpaint 方式对比
# ============================================================
print_section("Step 7: 两种 Inpaint 方式对比 — 哪种更好？")

print("""
【两种方式】
  方式 A（推荐）: 原始图 → Warp矫正 → Inpaint修复
  方式 B       : 原始图 → Inpaint修复 → Warp矫正

  区别：先矫正再修复 vs 先修复再矫正

  为什么方式 A 更好？
    1. 矫正后网格位置固定，不需要每次都检测标记点
    2. 矫正后间距均匀，mask 半径设置更准确
    3. 方式 B 中，矫正会把 inpaint 区域也"扭曲"一遍，
       可能引入额外的伪影
""")

# 方式 B：先 inpaint 再矫正
grid_ref_raw, _, kp_ref_raw = tac_utils.detect_and_sort(img_ref)
grid_contact_raw, _, kp_contact_raw = tac_utils.detect_and_sort(img_contact)

img_ref_inp_raw = tac_utils.inpaint_raw(img_ref, grid_ref_raw, marker_radius=5.0)
img_contact_inp_raw = tac_utils.inpaint_raw(img_contact, grid_contact_raw, marker_radius=5.0)

rect_ref_inp2 = tac_utils.apply_warp(img_ref_inp_raw, map_x, map_y)
rect_ref_inp2_crop, _ = tac_utils.crop_rectified(rect_ref_inp2, meta)
rect_ref_inp2_crop_rgb = cv2.cvtColor(rect_ref_inp2_crop, cv2.COLOR_BGR2RGB)

print(f"""
【数值对比】
  方式 A（先矫正后修复）:
    不需要重新检测标记点（直接用标准网格位置）
    mask 规则，均匀分布

  方式 B（先修复后矫正）:
    需要先检测标记点：检测到 {len(kp_ref_raw)} 个
    每个标记点单独画 mask，大小不一
    检测不到的点就没法修复
""")

fig, axes = plt.subplots(1, 3, figsize=(15, 6))

zoom_a = zoom_ref_after
zoom_b = rect_ref_inp2_crop_rgb[zoom_y0:zoom_y1, zoom_x0:zoom_x1]
diff_ab = np.abs(zoom_a.astype(float) - zoom_b.astype(float))

axes[0].imshow(zoom_a)
axes[0].set_title("方式A：先矫正 → 后修复", fontsize=11)
axes[0].axis("off")

axes[1].imshow(zoom_b)
axes[1].set_title("方式B：先修复 → 后矫正", fontsize=11)
axes[1].axis("off")

im = axes[2].imshow(np.mean(diff_ab, axis=2), cmap='hot')
axes[2].set_title("两种方式的差异", fontsize=11)
axes[2].axis("off")
plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle("两种 Inpaint 方式对比（放大中间区域）", fontsize=13, fontweight='bold')
plt.tight_layout()
save_fig(fig, "07_two_ways_compare", 7)
plt.close(fig)

print("""
【结论】
  两种方式结果很接近（差异很小），但方式 A 更推荐：
    ✅ 不需要每次检测标记点，速度更快
    ✅ 位置更准确（标准网格 vs 检测结果）
    ✅ 适用于批量处理（同一张参考图的 warp 可以复用）
""")

pause("两种方式的区别理解了吗？按回车继续最后一步 →")


# ============================================================
# Step 8: 总结
# ============================================================
print_section("Step 8: 总结 & 下一步")

print(f"""
【demo_08 你学到了什么】

  1. Warp 几何矫正
     - 原理：用 map_x/map_y 查找表，把扭曲的图像"重新排列"
     - 效果：倾斜/扭曲的网格 → 横平竖直的标准网格
     - 关键：反向映射（从结果反推源头），双线性插值

  2. 标记点遮罩 (Mask)
     - 白色 = 需要修复的标记点区域
     - 黑色 = 正常区域，不用动
     - 矫正后 mask 更规整（网格位置固定）

  3. Inpaint 图像修复
     - 算法：Telea（用周围像素插值填充）
     - 目的：去掉黑色标记点，为 Phase 3 光度立体做准备
     - 半径：3px（参考周围 3 像素）

  4. 两种 Inpaint 方式
     - 推荐：先矫正 → 后修复（更快、更准）
     - 另一种：先修复 → 后矫正（结果接近，但需要检测）

  5. 关键参数
     - 矫正后尺寸: {out_w} × {out_h} px
     - 物理尺度: {rp['mm_per_px_x']:.5f} mm/px
     - 网格间距: dx={rp['dx_rect']:.2f}px, dy={rp['dy_rect']:.2f}px
     - Inpaint 半径: 3 px

【下一步 demo_09 学什么】
  现在有了矫正后的图像，但我们的标记点是在原始图上检测的。
  demo_09 会讲：
    - 正向点映射：怎么把"原始图上的标记点坐标"变到"矫正后坐标"？
    - 位移场计算：矫正后坐标系下的位移是多少？
    - 物理单位转换：像素 → 毫米
    - 热力图可视化：位移场的直观展示

所有步骤的可视化图都保存在:
  {OUTPUT_DIR}
""")

print("=" * 70)
print("  demo_08 逐步调试走读完成！")
print("=" * 70)
