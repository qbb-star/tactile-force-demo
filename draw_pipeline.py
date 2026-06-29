"""
draw_pipeline.py
绘制视触觉传感器算法完整链路框图，适合PPT使用。
高分辨率、配色专业、适合学术汇报。
"""

import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ── 配色方案 ──────────────────────────────────────────
COLORS = {
    'p1_fill': '#E8F5E9',   # Phase 1 浅绿
    'p1_edge': '#2E7D32',   # Phase 1 深绿
    'p2_fill': '#E3F2FD',   # Phase 2 浅蓝
    'p2_edge': '#1565C0',   # Phase 2 深蓝
    'p3_fill': '#F5F5F5',   # Phase 3 浅灰
    'p3_edge': '#616161',   # Phase 3 深灰
    'p4_fill': '#F5F5F5',   # Phase 4 浅灰
    'p4_edge': '#616161',   # Phase 4 深灰
    'p5_fill': '#F5F5F5',   # Phase 5 浅灰
    'p5_edge': '#616161',   # Phase 5 深灰
    'title_bg_p1': '#2E7D32',
    'title_bg_p2': '#1565C0',
    'title_bg_p3': '#757575',
    'title_bg_p4': '#757575',
    'title_bg_p5': '#757575',
    'module_fill': 'white',
    'arrow_p1': '#2E7D32',
    'arrow_p2': '#1565C0',
    'arrow_p3': '#757575',
    'arrow_p4': '#757575',
    'text_dark': '#212121',
    'text_gray': '#757575',
    'goal_bg': '#FFF8E1',
    'goal_border': '#F9A825',
}

# ── 画布设置 ──────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(22, 9), dpi=150)
ax.set_xlim(0, 22)
ax.set_ylim(0, 8.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# 标题
ax.text(11, 8.0, '视触觉传感器算法完整链路', fontsize=22, fontweight='bold',
        ha='center', va='center', color=COLORS['text_dark'])
ax.text(11, 7.5, 'Visual Tactile Sensor Algorithm Pipeline', fontsize=11,
        ha='center', va='center', color=COLORS['text_gray'], style='italic')


# ── Phase 定义 ───────────────────────────────────────
def draw_phase(ax, x, y, w, h, title, status, modules, title_bg, body_fill, edge_color):
    """绘制单个Phase模块"""
    # 主体框
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=body_fill,
        edgecolor=edge_color,
        linewidth=2,
    )
    ax.add_patch(box)

    # 标题背景条
    title_h = 0.85
    title_bar = FancyBboxPatch(
        (x + 0.08, y + h - title_h),
        w - 0.16, title_h - 0.08,
        boxstyle="round,pad=0.03",
        facecolor=title_bg,
        edgecolor='none',
    )
    ax.add_patch(title_bar)

    # 标题文字
    ax.text(x + w/2, y + h - title_h/2,
            title, fontsize=11, fontweight='bold',
            ha='center', va='center', color='white')

    # 状态文字
    ax.text(x + w/2, y + h - title_h - 0.3,
            status, fontsize=8.5,
            ha='center', va='center', color=edge_color, fontweight='bold')

    # 模块
    mod_y_start = y + h - title_h - 0.75
    mod_h = 0.42
    mod_gap = 0.12
    for i, mod in enumerate(modules):
        my = mod_y_start - i * (mod_h + mod_gap)
        mod_box = FancyBboxPatch(
            (x + 0.25, my - mod_h/2),
            w - 0.5, mod_h,
            boxstyle="round,pad=0.02",
            facecolor=COLORS['module_fill'],
            edgecolor=edge_color,
            linewidth=1.2,
        )
        ax.add_patch(mod_box)
        ax.text(x + w/2, my,
                mod, fontsize=9, ha='center', va='center',
                color=COLORS['text_dark'])


# Phase 参数
pw = 3.5   # Phase 宽度
ph = 4.0   # Phase 高度
gap = 0.75 # Phase 间距
start_x = 0.8
start_y = 2.0

# Phase 1
draw_phase(ax, start_x, start_y, pw, ph,
           'Phase 1\n几何矫正',
           '[Done] 已完成',
           ['DoG 高斯差分检测', 'SimpleBlobDetector', 'PCA 主方向分析',
            'k-means 网格聚类', '三角剖分 Warp'],
           COLORS['title_bg_p1'], COLORS['p1_fill'], COLORS['p1_edge'])

# Phase 2
p2_x = start_x + pw + gap
draw_phase(ax, p2_x, start_y, pw, ph,
           'Phase 2\n位移计算 & 标记修复',
           '[Now] 进行中',
           ['参考/接触图配对', '逐点位移场计算', '位移向量/热力图',
            'Inpaint 标记点修复'],
           COLORS['title_bg_p2'], COLORS['p2_fill'], COLORS['p2_edge'])

# Phase 3
p3_x = p2_x + pw + gap
draw_phase(ax, p3_x, start_y, pw, ph,
           'Phase 3\n光度立体 & 法向估计',
           '[Todo] 待开始',
           ['三色光照分解 (R/G/B)', 'MLP 法向预测网络',
            '表面法向图 (Nx,Ny,Nz)'],
           COLORS['title_bg_p3'], COLORS['p3_fill'], COLORS['p3_edge'])

# Phase 4
p4_x = p3_x + pw + gap
draw_phase(ax, p4_x, start_y, pw, ph,
           'Phase 4\n深度重建',
           '[Todo] 待开始',
           ['法向量积分', 'Poisson/Dirichlet 求解',
            '深度图 (单位: mm)'],
           COLORS['title_bg_p4'], COLORS['p4_fill'], COLORS['p4_edge'])

# Phase 5
p5_x = p4_x + pw + gap
draw_phase(ax, p5_x, start_y, pw, ph,
           'Phase 5\n三维力估计',
           '[Todo] 待开始',
           ['接触力学模型', 'Hertz/FEM 接触理论',
            '三维力/力矩'],
           COLORS['title_bg_p5'], COLORS['p5_fill'], COLORS['p5_edge'])


# ── 箭头和中间输出 ────────────────────────────────────
def draw_arrow_with_label(ax, x1, y, x2, label, color):
    """绘制箭头+标签"""
    arrow = FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle='->,head_width=0.2,head_length=0.25',
        color=color, linewidth=2.5,
        mutation_scale=18,
    )
    ax.add_patch(arrow)
    # 标签
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, y + 0.35, label,
            fontsize=7.5, ha='center', va='bottom',
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      edgecolor=color, linewidth=1, alpha=0.95))


arrow_y = start_y + ph/2 - 0.3
draw_arrow_with_label(ax, start_x + pw + 0.05, arrow_y, p2_x - 0.05,
                      '矫正图像', COLORS['arrow_p1'])
draw_arrow_with_label(ax, p2_x + pw + 0.05, arrow_y, p3_x - 0.05,
                      '位移场 +\n无标记点图像', COLORS['arrow_p2'])
draw_arrow_with_label(ax, p3_x + pw + 0.05, arrow_y, p4_x - 0.05,
                      '表面法向图', COLORS['arrow_p3'])
draw_arrow_with_label(ax, p4_x + pw + 0.05, arrow_y, p5_x - 0.05,
                      '深度图', COLORS['arrow_p4'])


# ── 输入/输出标注 ─────────────────────────────────────
# 最左边输入
ax.annotate('原始图像',
            xy=(start_x, arrow_y), xytext=(start_x - 0.55, arrow_y),
            fontsize=9, ha='right', va='center',
            color=COLORS['text_dark'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['text_gray'], lw=1.5))

# 最右边输出
ax.annotate('Fx, Fy, Fz\nMx, My, Mz',
            xy=(p5_x + pw, arrow_y), xytext=(p5_x + pw + 0.65, arrow_y),
            fontsize=9, ha='left', va='center',
            color='#D84315', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['text_gray'], lw=1.5))


# ── 图例 ──────────────────────────────────────────────
legend_y = 1.1
legend_items = [
    ('[Done] 已完成', COLORS['title_bg_p1']),
    ('[Now] 进行中', COLORS['title_bg_p2']),
    ('[Todo] 待开始', COLORS['title_bg_p3']),
]
lx_start = 7.5
for i, (label, color) in enumerate(legend_items):
    lx = lx_start + i * 2.5
    rect = FancyBboxPatch((lx - 0.15, legend_y - 0.15), 0.3, 0.3,
                          boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor='none')
    ax.add_patch(rect)
    ax.text(lx + 0.25, legend_y, label, fontsize=9,
            ha='left', va='center', color=COLORS['text_dark'])


# ── 最终目标框 ───────────────────────────────────────
goal_box = FancyBboxPatch(
    (5.5, 0.1), 11, 0.6,
    boxstyle="round,pad=0.1",
    facecolor=COLORS['goal_bg'],
    edgecolor=COLORS['goal_border'],
    linewidth=2,
)
ax.add_patch(goal_box)
ax.text(11, 0.4,
        '最终目标：基于三色LED照明 + 点阵标记的视触觉图像，实现三维力/力矩估计',
        fontsize=11, ha='center', va='center',
        color='#E65100', fontweight='bold')


# ── 保存 ──────────────────────────────────────────────
plt.tight_layout(pad=0.5)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "algorithm_pipeline.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.3)
print(f"框图已保存: {out_path}")
print(f"图片尺寸: 22x9 英寸 @ 200dpi = 4400x1800 像素（适合PPT）")