"""
demo_07_step_by_step.py
Phase 2: 标记点跟踪 — 逐步调试走读版
每一步都有：目的讲解 + 数值统计 + 可视化输出
做到"数形结合"，帮助深入理解每一步在做什么

使用方法：
  运行后，每一步都会暂停，按回车继续下一步
  每一步的可视化图都会保存到 outputs/debug_demo07/
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

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "debug_demo07")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
print_section("Step 0: demo_07 整体路线总览")
print("""
demo_07 是 Phase 2 的入门，核心回答一个问题：
  「按压前后，标记点怎么动的？」

完整流程（6大步）：
  ① 加载图像       → 拿到参考图 + 接触图
  ② DoG斑点检测     → 从图里找出所有标记点（像素坐标）
  ③ 网格排序        → 把散乱的点排列成 21行×16列 的网格
  ④ 双图对比        → 参考图 vs 接触图，网格点摆在一起看
  ⑤ 叠加位移向量    → 同一坐标上画箭头，看每个点往哪动、动多少
  ⑥ 统计分析        → 位移多大？哪动得最厉害？

每一步都有「数」（数值统计）和「形」（可视化图），
做到数形结合，既要懂公式，也要懂图像。
""")
pause("准备好了吗？按回车开始 Step 1 →")


# ============================================================
# Step 1: 加载图像
# ============================================================
print_section("Step 1: 加载图像 — 拿到原始数据")

ref_path = os.path.join(DATA_DIR, REF_FILE)
contact_path = os.path.join(DATA_DIR, CONTACT_FILE)

img_ref = cv2.imread(ref_path, cv2.IMREAD_COLOR)
img_contact = cv2.imread(contact_path, cv2.IMREAD_COLOR)

print(f"""
【这一步在做什么】
  从磁盘读取两张图片：
    - ref.jpg        : 参考图像（没有按压，皮肤是平的）
    - contact.jpg    : 接触图像（按压后，皮肤变形了）

【数值信息】
  参考图像尺寸: {img_ref.shape[0]} × {img_ref.shape[1]} 像素 (高×宽)
  接触图像尺寸: {img_contact.shape[0]} × {img_contact.shape[1]} 像素
  图像通道数  : {img_ref.shape[2]} (BGR三通道)
  数据类型    : {img_ref.dtype} (uint8, 0-255)

【为什么要参考图？】
  位移是「相对量」——必须知道"没变形时长什么样"，
  才能算出"变形后动了多少"。
  就像尺子要有零刻度，才能量长度。
""")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
axes[0].set_title("参考图像 (Reference) — 无接触", fontsize=14)
axes[0].axis("off")
axes[1].imshow(cv2.cvtColor(img_contact, cv2.COLOR_BGR2RGB))
axes[1].set_title("接触图像 (Contact) — 按压后", fontsize=14)
axes[1].axis("off")
plt.tight_layout()
save_fig(fig, "01_load_images", 1)
plt.close(fig)

pause("看过图了吗？按回车继续 Step 2 →")


# ============================================================
# Step 2: DoG 斑点检测
# ============================================================
print_section("Step 2: DoG 斑点检测 — 找出所有标记点")

print("""
【这一步在做什么】
  用 Difference of Gaussian (DoG) 算法检测图像中的圆形斑点。
  原理：两张不同模糊程度的图相减，突出不同尺度的斑点。
  我们的标记点就是一个个小圆点，DoG 正好擅长找这个。

  调用的是 tac_utils.detect_blobs()，和 Phase 1 完全一样。
""")

kp_ref, dog_ref = tac_utils.detect_blobs(img_ref)
kp_contact, dog_contact = tac_utils.detect_blobs(img_contact)

print(f"""
【数值信息】
  参考图像检测到  : {len(kp_ref)} 个标记点
  接触图像检测到  : {len(kp_contact)} 个标记点
  理论应有        : {N_EXPECTED} 个 (21行 × 16列)
  参考图缺失      : {N_EXPECTED - len(kp_ref)} 个
  接触图缺失      : {N_EXPECTED - len(kp_contact)} 个

  检测到的点都是什么格式？举个例子：
    第1个点: 坐标 ({kp_ref[0].pt[0]:.1f}, {kp_ref[0].pt[1]:.1f}) px
             大小 {kp_ref[0].size:.1f} px (斑点直径)

【直观理解】
  每个 keypoint 就是一个「我找到一个圆点，在这里」的记录。
  但现在这些点是「乱序」的——你不知道哪个点是第几行第几列。
  下一步排序就是解决这个问题。
""")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0, 0].imshow(dog_ref, cmap='gray')
axes[0, 0].set_title(f"参考图 DoG 响应 (检测到 {len(kp_ref)} 个点)", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
for kp in kp_ref:
    circle = plt.Circle(kp.pt, kp.size/2, color='lime', fill=False, linewidth=1)
    axes[0, 1].add_patch(circle)
axes[0, 1].set_title("参考图检测结果（绿圈=标记点）", fontsize=12)
axes[0, 1].axis("off")

axes[1, 0].imshow(dog_contact, cmap='gray')
axes[1, 0].set_title(f"接触图 DoG 响应 (检测到 {len(kp_contact)} 个点)", fontsize=12)
axes[1, 0].axis("off")

axes[1, 1].imshow(cv2.cvtColor(img_contact, cv2.COLOR_BGR2RGB))
for kp in kp_contact:
    circle = plt.Circle(kp.pt, kp.size/2, color='lime', fill=False, linewidth=1)
    axes[1, 1].add_patch(circle)
axes[1, 1].set_title("接触图检测结果（绿圈=标记点）", fontsize=12)
axes[1, 1].axis("off")

plt.tight_layout()
save_fig(fig, "02_dog_detection", 2)
plt.close(fig)

pause("理解DoG检测了吗？按回车继续 Step 3 →")


# ============================================================
# Step 3: PCA + 网格排序
# ============================================================
print_section("Step 3: 网格排序 — 把散乱的点排成 21×16 的网格")

print("""
【这一步在做什么】
  输入：一堆散乱的点坐标（只有(x,y)，不知道谁是谁）
  输出：21行 × 16列 的网格矩阵 grid[r, c] = (x, y)

  核心算法（和 Phase 1 一致）：
    1. PCA 主成分分析 → 找到网格的"行方向"和"列方向"
    2. k-means 聚类   → 把点按行和列分组
    3. 贪心分配       → 每个点落到最近的网格单元
    4. 多策略择优     → 8种方向组合+rank-based兜底，选拓扑最好的

  调用的是 tac_utils.sort_grid()
""")

grid_ref = tac_utils.sort_grid(kp_ref)
grid_contact = tac_utils.sort_grid(kp_contact)

n_valid_ref = int(np.sum(~np.isnan(grid_ref[:, :, 0])))
n_valid_contact = int(np.sum(~np.isnan(grid_contact[:, :, 0])))
n_empty_ref = N_EXPECTED - n_valid_ref
n_empty_contact = N_EXPECTED - n_valid_contact

print(f"""
【数值信息】
  参考图排序后有效点: {n_valid_ref} / {N_EXPECTED} (空 {n_empty_ref} 个)
  接触图排序后有效点: {n_valid_contact} / {N_EXPECTED} (空 {n_empty_contact} 个)

  举几个网格坐标的例子：
    grid_ref[0, 0]   = ({grid_ref[0, 0, 0]:.1f}, {grid_ref[0, 0, 1]:.1f})   第1行第1列 (左上角)
    grid_ref[0, 15]  = ({grid_ref[0, -1, 0]:.1f}, {grid_ref[0, -1, 1]:.1f})   第1行最后1列 (右上角)
    grid_ref[10, 8]  = ({grid_ref[10, 8, 0]:.1f}, {grid_ref[10, 8, 1]:.1f})  中间位置

  行间距大概是多少？
    第0行y坐标: {np.nanmean(grid_ref[0, :, 1]):.1f} px
    第10行y坐标: {np.nanmean(grid_ref[10, :, 1]):.1f} px
    第20行y坐标: {np.nanmean(grid_ref[-1, :, 1]):.1f} px
    平均行间距: {(np.nanmean(grid_ref[-1, :, 1]) - np.nanmean(grid_ref[0, :, 1])) / (N_ROWS - 1):.1f} px

  列间距大概是多少？
    第0列x坐标: {np.nanmean(grid_ref[:, 0, 0]):.1f} px
    第15列x坐标: {np.nanmean(grid_ref[:, -1, 0]):.1f} px
    平均列间距: {(np.nanmean(grid_ref[:, -1, 0]) - np.nanmean(grid_ref[:, 0, 0])) / (N_COLS - 1):.1f} px
""")

print("""
【拓扑检查】
  排序质量怎么样？用拓扑评分来衡量（0=完美，越大越乱）：
""")
score_ref = tac_utils.grid_topology_score(grid_ref)
score_contact = tac_utils.grid_topology_score(grid_contact)
_, issues_ref = tac_utils.validate_grid_topology(grid_ref)
_, issues_contact = tac_utils.validate_grid_topology(grid_contact)
print(f"    参考图拓扑评分: {score_ref} 个问题  {'✅ 完美' if score_ref == 0 else '❌ 有问题'}")
print(f"    接触图拓扑评分: {score_contact} 个问题  {'✅ 完美' if score_contact == 0 else '❌ 有问题'}")
if issues_ref:
    print(f"    参考图问题: {issues_ref}")
if issues_contact:
    print(f"    接触图问题: {issues_contact}")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
for r in range(N_ROWS):
    xs = [grid_ref[r, c, 0] for c in range(N_COLS) if not np.isnan(grid_ref[r, c, 0])]
    ys = [grid_ref[r, c, 1] for c in range(N_COLS) if not np.isnan(grid_ref[r, c, 0])]
    if len(xs) > 1:
        axes[0].plot(xs, ys, 'lime', linewidth=0.8, alpha=0.7)
for c in range(N_COLS):
    xs = [grid_ref[r, c, 0] for r in range(N_ROWS) if not np.isnan(grid_ref[r, c, 0])]
    ys = [grid_ref[r, c, 1] for r in range(N_ROWS) if not np.isnan(grid_ref[r, c, 0])]
    if len(xs) > 1:
        axes[0].plot(xs, ys, 'lime', linewidth=0.8, alpha=0.7)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_ref[r, c, 0]):
            axes[0].plot(grid_ref[r, c, 0], grid_ref[r, c, 1], 'o', ms=4, color='lime')
axes[0].set_title(f"参考图排序结果（网格线）  有效点={n_valid_ref}/{N_EXPECTED}", fontsize=13)
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(img_contact, cv2.COLOR_BGR2RGB))
for r in range(N_ROWS):
    xs = [grid_contact[r, c, 0] for c in range(N_COLS) if not np.isnan(grid_contact[r, c, 0])]
    ys = [grid_contact[r, c, 1] for c in range(N_COLS) if not np.isnan(grid_contact[r, c, 0])]
    if len(xs) > 1:
        axes[1].plot(xs, ys, 'lime', linewidth=0.8, alpha=0.7)
for c in range(N_COLS):
    xs = [grid_contact[r, c, 0] for r in range(N_ROWS) if not np.isnan(grid_contact[r, c, 0])]
    ys = [grid_contact[r, c, 1] for r in range(N_ROWS) if not np.isnan(grid_contact[r, c, 0])]
    if len(xs) > 1:
        axes[1].plot(xs, ys, 'lime', linewidth=0.8, alpha=0.7)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_contact[r, c, 0]):
            axes[1].plot(grid_contact[r, c, 0], grid_contact[r, c, 1], 'o', ms=4, color='lime')
axes[1].set_title(f"接触图排序结果（网格线）  有效点={n_valid_contact}/{N_EXPECTED}", fontsize=13)
axes[1].axis("off")

plt.tight_layout()
save_fig(fig, "03_grid_sorted", 3)
plt.close(fig)

pause("理解网格排序了吗？按回车继续 Step 4 →")


# ============================================================
# Step 4: 双图对比 — 并排看
# ============================================================
print_section("Step 4: 双图对比 — 并排看参考图 vs 接触图")

print("""
【这一步在做什么】
  把两张图的网格放在一起对比，直观感受"哪里变形了"。
  上排是参考图（没按压），下排是接触图（按压后）。

  左到右：原图 → 点标记 → 网格连线
  越往右，信息越抽象，但结构越清晰。
""")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 第1行：参考图像
img_rgb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
axes[0, 0].imshow(img_rgb_ref)
axes[0, 0].set_title("参考图像", fontsize=12)
axes[0, 0].axis("off")

axes[0, 1].imshow(img_rgb_ref)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_ref[r, c, 0]):
            axes[0, 1].plot(grid_ref[r, c, 0], grid_ref[r, c, 1], 'o', ms=3, color='lime')
axes[0, 1].set_title(f"参考网格点 ({n_valid_ref}个)", fontsize=12)
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
axes[0, 2].set_title("参考网格连线", fontsize=12)
axes[0, 2].axis("off")

# 第2行：接触图像
img_rgb_contact = cv2.cvtColor(img_contact, cv2.COLOR_BGR2RGB)
axes[1, 0].imshow(img_rgb_contact)
axes[1, 0].set_title("接触图像", fontsize=12)
axes[1, 0].axis("off")

axes[1, 1].imshow(img_rgb_contact)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_contact[r, c, 0]):
            axes[1, 1].plot(grid_contact[r, c, 0], grid_contact[r, c, 1], 'o', ms=3, color='lime')
axes[1, 1].set_title(f"接触网格点 ({n_valid_contact}个)", fontsize=12)
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
axes[1, 2].set_title("接触网格连线", fontsize=12)
axes[1, 2].axis("off")

plt.tight_layout()
save_fig(fig, "04_two_image_compare", 4)
plt.close(fig)

print(f"""
【怎么看这张图】
  上下对比看：
    - 接触区（下方偏右）的网格是不是被"挤"了？
    - 远离接触区的地方，网格是不是还保持整齐？
  左右对比看：
    - 从"点"到"线"，信息密度降低了，但结构更清晰了
    - 连线能帮你快速看出"行是不是平的、列是不是直的"
""")

pause("双图对比看清楚了吗？按回车继续 Step 5 →")


# ============================================================
# Step 5: 叠加位移向量
# ============================================================
print_section("Step 5: 位移向量可视化 — 每个点往哪动？动多少？")

print("""
【这一步在做什么】
  位移 = 接触位置 - 参考位置
    dx = grid_contact[r,c,0] - grid_ref[r,c,0]
    dy = grid_contact[r,c,1] - grid_ref[r,c,1]

  只有「同一个网格单元」的两个点才能相减
  （这就是为什么排序很重要——排错了就减错了）
""")

displacements = []
valid_pairs = []
dx_all = np.full((N_ROWS, N_COLS), np.nan)
dy_all = np.full((N_ROWS, N_COLS), np.nan)
disp_mag = np.full((N_ROWS, N_COLS), np.nan)

for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_ref[r, c, 0]) and not np.isnan(grid_contact[r, c, 0]):
            dx = grid_contact[r, c, 0] - grid_ref[r, c, 0]
            dy = grid_contact[r, c, 1] - grid_ref[r, c, 1]
            dx_all[r, c] = dx
            dy_all[r, c] = dy
            mag = np.sqrt(dx**2 + dy**2)
            disp_mag[r, c] = mag
            displacements.append(mag)
            valid_pairs.append((r, c))

displacements = np.array(displacements)
n_valid = len(displacements)

print(f"""
【数值信息】
  有效配对数: {n_valid} / {N_EXPECTED}
  位移统计（像素）:
    均值  : {displacements.mean():.3f} px
    中位数: {np.median(displacements):.3f} px
    最大  : {displacements.max():.3f} px
    最小  : {displacements.min():.3f} px
    标准差: {displacements.std():.3f} px

  dx (水平位移) 范围: [{np.nanmin(dx_all):.3f}, {np.nanmax(dx_all):.3f}] px
  dy (垂直位移) 范围: [{np.nanmin(dy_all):.3f}, {np.nanmax(dy_all):.3f}] px

  位移最大的 5 个点:
""")
top_idx = np.argsort(displacements)[::-1][:5]
for i, idx in enumerate(top_idx):
    r, c = valid_pairs[idx]
    dx = dx_all[r, c]
    dy = dy_all[r, c]
    print(f"    #{i+1}  [{r:2d},{c:2d}]  |d|={displacements[idx]:.3f}px  "
          f"dx={dx:+.3f}  dy={dy:+.3f}")

print(f"""
【直观理解】
  每个箭头 = 「红点（参考位置）→ 绿点（接触位置）」的移动
  箭头长度 = 位移大小
  箭头方向 = 位移方向

  箭头放大了 2 倍，不然太小看不清。
""")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# 左图：红点=参考，绿点=接触
axes[0].imshow(img_rgb_contact)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_ref[r, c, 0]):
            axes[0].plot(grid_ref[r, c, 0], grid_ref[r, c, 1], 'o', ms=4, color='red', alpha=0.6)
        if not np.isnan(grid_contact[r, c, 0]):
            axes[0].plot(grid_contact[r, c, 0], grid_contact[r, c, 1], 'o', ms=4, color='lime', alpha=0.6)
axes[0].set_title("红=参考位置  绿=接触位置", fontsize=13)
axes[0].axis("off")

# 右图：位移箭头（放大2倍）
axes[1].imshow(img_rgb_contact)
for r in range(N_ROWS):
    for c in range(N_COLS):
        if not np.isnan(grid_ref[r, c, 0]) and not np.isnan(grid_contact[r, c, 0]):
            dx = grid_contact[r, c, 0] - grid_ref[r, c, 0]
            dy = grid_contact[r, c, 1] - grid_ref[r, c, 1]
            axes[1].arrow(grid_ref[r, c, 0], grid_ref[r, c, 1],
                          dx * 2, dy * 2,
                          head_width=2, head_length=3, fc='yellow', ec='yellow',
                          alpha=0.7, linewidth=0.5)
axes[1].set_title(f"位移向量（2倍放大）  n={n_valid}  mean={displacements.mean():.2f}px  max={displacements.max():.2f}px", fontsize=12)
axes[1].axis("off")

plt.tight_layout()
save_fig(fig, "05_displacement_vectors", 5)
plt.close(fig)

pause("位移向量理解了吗？按回车继续 Step 6 →")


# ============================================================
# Step 6: 位移热力图
# ============================================================
print_section("Step 6: 位移热力图 — 哪里变形最严重？")

print("""
【这一步在做什么】
  把每个网格点的位移大小 |d| = sqrt(dx² + dy²) 画成热力图。
  颜色越亮（黄白）= 位移越大
  颜色越暗（黑红）= 位移越小

  这是接触力最直观的可视化——压得越用力，变形越大。
""")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# |d| 热力图
im0 = axes[0].imshow(disp_mag, cmap='hot', origin='upper')
axes[0].set_title("位移大小 |d| (热力图)", fontsize=13)
axes[0].set_xlabel("列号")
axes[0].set_ylabel("行号")
plt.colorbar(im0, ax=axes[0], label='|d| (px)')

# dx 热力图
im1 = axes[1].imshow(dx_all, cmap='RdBu_r', origin='upper',
                     vmin=-np.nanmax(np.abs(dx_all)), vmax=np.nanmax(np.abs(dx_all)))
axes[1].set_title("水平位移 dx (红=向右，蓝=向左)", fontsize=13)
axes[1].set_xlabel("列号")
axes[1].set_ylabel("行号")
plt.colorbar(im1, ax=axes[1], label='dx (px)')

# dy 热力图
im2 = axes[2].imshow(dy_all, cmap='RdBu_r', origin='upper',
                     vmin=-np.nanmax(np.abs(dy_all)), vmax=np.nanmax(np.abs(dy_all)))
axes[2].set_title("垂直位移 dy (红=向下，蓝=向上)", fontsize=13)
axes[2].set_xlabel("列号")
axes[2].set_ylabel("行号")
plt.colorbar(im2, ax=axes[2], label='dy (px)')

plt.tight_layout()
save_fig(fig, "06_displacement_heatmap", 6)
plt.close(fig)

print(f"""
【怎么看这张图】
  左图 |d|: 最亮的地方就是变形最严重的地方，也就是接触中心
  中图 dx: 正值=点向右移，负值=向左移
  右图 dy: 正值=点向下移，负值=向上移

  物理直觉：
    如果从上方按压，皮肤会向四周"挤"，
    接触点上方的点往上移，下方的点往下移，
    左边的往左移，右边的往右移。
    你在图里看到这个规律了吗？
""")

pause("热力图理解了吗？按回车继续最后一步 →")


# ============================================================
# Step 7: 总结
# ============================================================
print_section("Step 7: 总结 & 下一步")

print(f"""
【demo_07 你学到了什么】

  1. 数据来源：参考图（无变形）+ 接触图（变形后）
  2. 检测标记点：DoG 算法找圆点，参考图 {len(kp_ref)} 个，接触图 {len(kp_contact)} 个
  3. 网格排序：散乱点 → 21×16 网格，多策略择优保证鲁棒性
  4. 位移计算：grid_contact - grid_ref = 位移场
  5. 可视化：
     - 散点图：看每个点的位置
     - 箭头图：看每个点的移动方向和大小
     - 热力图：看整体变形分布

  有效配对: {n_valid} / {N_EXPECTED} 个点
  最大位移: {displacements.max():.3f} px
  平均位移: {displacements.mean():.3f} px

【下一步 demo_08 学什么】
  现在我们算的位移是「原始图像坐标系」下的像素位移。
  但我们需要的是「矫正后坐标系」下的物理位移（mm）。

  demo_08 会讲：
    - 图像矫正（Warp）：把倾斜的图像"掰正"
    - Inpaint 修复：把标记点的位置补成皮肤纹理
    - 正向点映射：原始坐标 → 矫正后坐标

所有步骤的可视化图都保存在:
  {OUTPUT_DIR}
""")

print("=" * 70)
print("  demo_07 逐步调试走读完成！")
print("=" * 70)
