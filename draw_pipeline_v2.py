"""
draw_pipeline_v2.py
算法链路框图 — PPT 大字版
特点：
  - 无标题（节省空间）
  - 字体放大（缩放后仍清晰）
  - Phase 1/2 ✅ 已完成，Phase 3 🔄 进行中
"""

import os
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

COLORS = {
    'done_fill': '#E8F5E9',
    'done_edge': '#2E7D32',
    'doing_fill': '#E3F2FD',
    'doing_edge': '#1565C0',
    'todo_fill': '#F5F5F5',
    'todo_edge': '#616161',
    'title_done': '#2E7D32',
    'title_doing': '#1565C0',
    'title_todo': '#757575',
    'module_fill': 'white',
    'text_dark': '#212121',
    'text_gray': '#757575',
}

fig, ax = plt.subplots(1, 1, figsize=(22, 7.5), dpi=200)
ax.set_xlim(0, 22)
ax.set_ylim(0, 7.5)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')


def draw_phase(ax, x, y, w, h, phase_name, title_cn, status, modules, title_bg, body_fill, edge_color):
    """绘制单个 Phase 模块（大字版）"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.06",
        facecolor=body_fill,
        edgecolor=edge_color,
        linewidth=2.5,
    )
    ax.add_patch(box)

    title_h = 0.95
    title_bar = FancyBboxPatch(
        (x + 0.1, y + h - title_h),
        w - 0.2, title_h - 0.1,
        boxstyle="round,pad=0.04",
        facecolor=title_bg,
        edgecolor='none',
    )
    ax.add_patch(title_bar)

    ax.text(x + w/2, y + h - title_h/2,
            phase_name, fontsize=14, fontweight='bold',
            ha='center', va='center', color='white')

    ax.text(x + w/2, y + h - title_h - 0.32,
            status, fontsize=11,
            ha='center', va='center', color=edge_color, fontweight='bold')

    mod_y_start = y + h - title_h - 0.85
    mod_h = 0.5
    mod_gap = 0.14
    for i, mod in enumerate(modules):
        my = mod_y_start - i * (mod_h + mod_gap)
        mod_box = FancyBboxPatch(
            (x + 0.3, my - mod_h/2),
            w - 0.6, mod_h,
            boxstyle="round,pad=0.03",
            facecolor=COLORS['module_fill'],
            edgecolor=edge_color,
            linewidth=1.5,
        )
        ax.add_patch(mod_box)
        ax.text(x + w/2, my,
                mod, fontsize=11, ha='center', va='center',
                color=COLORS['text_dark'])


pw = 3.4
ph = 5.0
gap = 0.55
start_x = 1.2
start_y = 1.6

draw_phase(ax, start_x, start_y, pw, ph,
           'Phase 1', '几何矫正',
           '[已完成]',
           ['DoG 高斯差分检测', 'SimpleBlobDetector', 'PCA 主方向分析',
            'k-means 网格聚类', '三角剖分 Warp'],
           COLORS['title_done'], COLORS['done_fill'], COLORS['done_edge'])

p2_x = start_x + pw + gap
draw_phase(ax, p2_x, start_y, pw, ph,
           'Phase 2', '位移计算 & 标记修复',
           '[已完成]',
           ['参考/接触图配对', '正向点映射', '位移场计算 (mm)',
            'Inpaint 标记点修复'],
           COLORS['title_done'], COLORS['done_fill'], COLORS['done_edge'])

p3_x = p2_x + pw + gap
draw_phase(ax, p3_x, start_y, pw, ph,
           'Phase 3', '光度立体 & 法向估计',
           '[进行中]',
           ['三色光照分解 R/G/B', '法向预测网络',
            '表面法向图 (Nx,Ny,Nz)'],
           COLORS['title_doing'], COLORS['doing_fill'], COLORS['doing_edge'])

p4_x = p3_x + pw + gap
draw_phase(ax, p4_x, start_y, pw, ph,
           'Phase 4', '深度重建',
           '[待开始]',
           ['法向量积分', 'Poisson 求解',
            '深度图 (单位: mm)'],
           COLORS['title_todo'], COLORS['todo_fill'], COLORS['todo_edge'])

p5_x = p4_x + pw + gap
draw_phase(ax, p5_x, start_y, pw, ph,
           'Phase 5', '三维力估计',
           '[待开始]',
           ['接触力学模型', 'Hertz/FEM 理论',
            '三维力 / 力矩'],
           COLORS['title_todo'], COLORS['todo_fill'], COLORS['todo_edge'])


def draw_arrow_with_label(ax, x1, y, x2, label, color):
    arrow = FancyArrowPatch(
        (x1, y), (x2, y),
        arrowstyle='->,head_width=0.22,head_length=0.28',
        color=color, linewidth=3,
        mutation_scale=20,
    )
    ax.add_patch(arrow)
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, y + 0.38, label,
            fontsize=10, ha='center', va='bottom',
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white',
                      edgecolor=color, linewidth=1.2, alpha=0.95))


arrow_y = start_y + ph/2
draw_arrow_with_label(ax, start_x + pw + 0.05, arrow_y, p2_x - 0.05,
                      '矫正图像', COLORS['done_edge'])
draw_arrow_with_label(ax, p2_x + pw + 0.05, arrow_y, p3_x - 0.05,
                      '位移场 +\n无标记点图', COLORS['done_edge'])
draw_arrow_with_label(ax, p3_x + pw + 0.05, arrow_y, p4_x - 0.05,
                      '法向图', COLORS['doing_edge'])
draw_arrow_with_label(ax, p4_x + pw + 0.05, arrow_y, p5_x - 0.05,
                      '深度图', COLORS['todo_edge'])


ax.annotate('原始图像',
            xy=(start_x, arrow_y), xytext=(start_x - 0.8, arrow_y),
            fontsize=12, ha='right', va='center',
            color=COLORS['text_dark'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['text_gray'], lw=2))

ax.annotate('Fx, Fy, Fz\nMx, My, Mz',
            xy=(p5_x + pw, arrow_y), xytext=(p5_x + pw + 0.9, arrow_y),
            fontsize=12, ha='left', va='center',
            color='#D84315', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLORS['text_gray'], lw=2))


plt.tight_layout(pad=0.3)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "algorithm_pipeline_v2.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white', pad_inches=0.2)
print(f"框图已保存: {out_path}")
