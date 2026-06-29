"""
demo_01_read_image.py — 读取触觉图像并了解基本信息
====================================================
目的：熟悉触觉图像的原始形态，了解图像尺寸、色彩分布等基本信息。

触觉传感器背景：
  视触觉传感器（如 GelSight）通过相机拍摄软质凝胶表面的形变。
  凝胶表面印有点阵标记（marker dots），三色 LED 从不同角度打光。
  - 红色/绿色/蓝色通道分别对应不同方向的光照
  - 标记点是暗色圆点，在亮背景下清晰可见
  - 图像的 RGB 颜色分布编码了表面法向信息

关键差异（与旧 calibData 图像相比）：
  - 旧图像（calibData）：传感器横向放置，左侧有遮挡，网格 16列×21行
  - 新图像（data_wenli）：传感器旋转 90°，下方有遮挡，网格 16列×21行
  - 这意味着旧代码中的 mask_left_px 需改为 mask_bottom_px

运行：python demo_01_read_image.py
"""

import cv2
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os

# ============================================================
# 配置：触觉图像路径（使用无接触的参考帧 ref.jpg）
# ============================================================
DATA_DIR = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
IMAGE_FILE = "ref.jpg"  # 无接触参考帧，用于几何矫正
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 读取图像
# ============================================================
img_path = os.path.join(DATA_DIR, IMAGE_FILE)
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError(f"无法读取图像: {img_path}")

print("=" * 60)
print("【触觉图像基本信息】")
print(f"  文件: {IMAGE_FILE}")
print(f"  形状 (H, W, C): {img.shape}")
print(f"  高度: {img.shape[0]} px")
print(f"  宽度: {img.shape[1]} px")
print(f"  通道数: {img.shape[2]} (BGR)")
print(f"  数据类型: {img.dtype}")

# ============================================================
# 2. 分析各通道的统计信息
# ============================================================
# OpenCV 读取的是 BGR 格式，转换为 RGB 便于理解
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("\n【各通道统计信息 (RGB, 0-255)】")
channel_names = ["Red (R)", "Green (G)", "Blue (B)"]
for i, name in enumerate(channel_names):
    channel = img_rgb[:, :, i]
    print(f"  {name}: min={channel.min()}, max={channel.max()}, "
          f"mean={channel.mean():.1f}, std={channel.std():.1f}")

# ============================================================
# 3. 理解 RGB 三色光照原理
# ============================================================
print("\n【三色光照原理】")
print("  视触觉传感器的三色 LED 从不同角度照射凝胶表面：")
print("  - 红色 LED 通常从左侧照射 → R 通道编码左右方向的梯度")
print("  - 绿色 LED 通常从右侧照射 → G 通道编码另一侧梯度")
print("  - 蓝色 LED 通常从上方照射 → B 通道编码上下方向的梯度")
print("  - 表面法向可以通过 RGB 三通道的强度比来估计")
print("  - 标记点（暗色圆点）在全通道中都是暗的")

# ============================================================
# 4. 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原始图像（BGR → RGB 显示）
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("原始触觉图像 (RGB)", fontsize=12)
axes[0, 0].axis("off")

# R 通道
axes[0, 1].imshow(img_rgb[:, :, 0], cmap="Reds")
axes[0, 1].set_title("R 通道 (红色光照)", fontsize=12)
axes[0, 1].axis("off")

# G 通道
axes[0, 2].imshow(img_rgb[:, :, 1], cmap="Greens")
axes[0, 2].set_title("G 通道 (绿色光照)", fontsize=12)
axes[0, 2].axis("off")

# B 通道
axes[1, 0].imshow(img_rgb[:, :, 2], cmap="Blues")
axes[1, 0].set_title("B 通道 (蓝色光照)", fontsize=12)
axes[1, 0].axis("off")

# 灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
axes[1, 1].imshow(gray, cmap="gray")
axes[1, 1].set_title("灰度图", fontsize=12)
axes[1, 1].axis("off")

# 各通道直方图
for i, (name, color) in enumerate(zip(
    ["R", "G", "B"], ["red", "green", "blue"]
)):
    axes[1, 2].hist(
        img_rgb[:, :, i].ravel(), bins=256, range=(0, 256),
        color=color, alpha=0.5, label=name
    )
axes[1, 2].set_title("RGB 直方图", fontsize=12)
axes[1, 2].legend()
axes[1, 2].set_xlim(0, 255)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "demo_01_overview.png"), dpi=150)
plt.close()
print(f"\n[保存] 可视化结果已保存到 {OUTPUT_DIR}/demo_01_overview.png")

print("\n✅ demo_01 完成！现在你对触觉图像有了基本认识。")
print("  下一步: demo_02_dog_detect.py — 学习 DoG 标记点检测")