"""
demo_02_dog_detect.py — DoG（高斯差分）标记点检测
================================================
目的：理解 DoG 预处理算法如何从触觉图像中提取标记点（marker dots）。

算法原理：
  DoG (Difference of Gaussians) 是一种经典的 blob 检测预处理方法。
  
  步骤：
  1. 将彩色图像转为灰度图
  2. 用大核高斯模糊 → 保留低频背景（模糊掉标记点）
  3. 用小核高斯模糊 → 保留更多细节（包含标记点）
  4. 两者相减 → 突出与背景有差异的区域（即标记点变亮）
  5. 乘以增益 → 增强对比度
  6. OpenCV SimpleBlobDetector 检测圆形 blob

  数学表达：
    DoG = (G_large - G_small) * gain
    其中 G_large = GaussianBlur(gray, ksize_large)
          G_small = GaussianBlur(gray, ksize_small)

  为什么 DoG 有效？
    - 大核模糊相当于"只看背景"
    - 小核模糊相当于"背景+细节"
    - 两者相减得到的就是"细节"（标记点）
    - 标记点是暗色圆点，在 DoG 结果中呈现为亮斑

  注意：当前图像（data_wenli/ref.jpg）相比旧 calibData 图像旋转了 90°，
  网格在图像中呈现为竖长方形（16列×21行），底部有遮挡区域。

运行：python demo_02_dog_detect.py
"""

import cv2
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os

# ============================================================
# 配置
# ============================================================
DATA_DIR = r"D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli"
IMAGE_FILE = "ref.jpg"  # 无接触参考帧
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. 读取图像
# ============================================================
img_path = os.path.join(DATA_DIR, IMAGE_FILE)
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转为灰度图（DoG 在灰度域操作）
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

print("=" * 60)
print("【DoG 标记检测 — 逐步演示】")
print(f"  图像尺寸: {gray.shape}")

# ============================================================
# 2. 高斯模糊参数
# ============================================================
# 大核：模糊掉标记点，只保留背景
blur_large_ksize = 27  # 奇数
blur_small_ksize = 5   # 奇数，必须小于大核

# 计算 sigma（OpenCV 默认公式）
sigma_large = 0.3 * ((blur_large_ksize - 1) * 0.5 - 1) + 0.8
sigma_small = 0.3 * ((blur_small_ksize - 1) * 0.5 - 1) + 0.8

print(f"\n【高斯模糊参数】")
print(f"  大核: ksize={blur_large_ksize}, sigma≈{sigma_large:.1f}")
print(f"  小核: ksize={blur_small_ksize}, sigma≈{sigma_small:.1f}")

# 执行高斯模糊
blur_large = cv2.GaussianBlur(gray, (blur_large_ksize, blur_large_ksize), sigma_large)
blur_small = cv2.GaussianBlur(gray, (blur_small_ksize, blur_small_ksize), sigma_small)

# ============================================================
# 3. DoG 计算
# ============================================================
print("\n【DoG 计算】")
print(f"  DoG = (blur_large - blur_small) * gain")
print(f"  blur_large 保留了低频背景（标记点被模糊掉）")
print(f"  blur_small 保留了更多细节（包含标记点）")
print(f"  两者相减 → 标记点区域呈现为亮斑")

# 测试三个不同的 gain 值
gain_values = [5.0, 10.0, 15.0]
dog_results = []

for gain in gain_values:
    dog = (blur_large - blur_small) * gain
    dog_clipped = np.clip(dog, 0.0, 255.0).astype(np.uint8)
    dog_results.append(dog_clipped)
    print(f"  gain={gain:.1f}: min={dog_clipped.min()}, max={dog_clipped.max()}, "
          f"mean={dog_clipped.mean():.1f}")

# 使用默认 gain=12.9（与 viewtacv2 配置一致）
dog_gain = 12.9
dog = (blur_large - blur_small) * dog_gain
dog_u8 = np.clip(dog, 0.0, 255.0).astype(np.uint8)

# ============================================================
# 4. Blob 检测
# ============================================================
print("\n【Blob 检测】")
print(f"  使用 OpenCV SimpleBlobDetector 检测圆形 blob")

# 配置 blob 检测器参数
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 171.0  # 降低阈值，确保低对比度图像稳定检测
params.maxThreshold = 255.0
params.thresholdStep = 11.0
params.minDistBetweenBlobs = 6.0
params.filterByArea = True
params.minArea = 8.0
params.maxArea = 1500.0
params.filterByCircularity = True
params.minCircularity = 0.72
params.filterByConvexity = False
params.filterByInertia = False
params.blobColor = 255  # 检测亮斑

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(dog_u8)

# 按响应强度排序
keypoints = sorted(keypoints, key=lambda k: k.response, reverse=True)

print(f"  检测到 {len(keypoints)} 个关键点")
print(f"  期望点数: 16 × 21 = 336")
print(f"  最强点响应: {keypoints[0].response:.2f}" if keypoints else "  无检测结果")

# 在图像上绘制检测到的关键点
img_with_kps = cv2.drawKeypoints(
    img, keypoints, None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# ============================================================
# 5. 可视化
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# 原始图像
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original Image", fontsize=11)
axes[0, 0].axis("off")

# 灰度图
axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Grayscale", fontsize=11)
axes[0, 1].axis("off")

# 大核模糊
axes[0, 2].imshow(blur_large, cmap="gray")
axes[0, 2].set_title(f"Gaussian Blur (ksize={blur_large_ksize})", fontsize=11)
axes[0, 2].axis("off")

# 小核模糊
axes[0, 3].imshow(blur_small, cmap="gray")
axes[0, 3].set_title(f"Gaussian Blur (ksize={blur_small_ksize})", fontsize=11)
axes[0, 3].axis("off")

# 不同 gain 的 DoG 结果
for i, (gain, dog_img) in enumerate(zip(gain_values, dog_results)):
    axes[1, i].imshow(dog_img, cmap="hot")
    axes[1, i].set_title(f"DoG (gain={gain:.1f})", fontsize=11)
    axes[1, i].axis("off")

# 检测结果
img_with_kps_rgb = cv2.cvtColor(img_with_kps, cv2.COLOR_BGR2RGB)
axes[1, 3].imshow(img_with_kps_rgb)
axes[1, 3].set_title(f"Detected Keypoints: {len(keypoints)}", fontsize=11)
axes[1, 3].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "demo_02_dog_detect.png"), dpi=150)
plt.close()
print(f"\n[保存] 可视化结果已保存到 {OUTPUT_DIR}/demo_02_dog_detect.png")

# ============================================================
# 6. 总结
# ============================================================
print("\n" + "=" * 60)
print("【关键理解】")
print("  1. DoG 通过大核-小核的差值来突出标记点（暗色圆点）")
print("  2. gain 参数控制对比度：太大会引入噪声，太小会丢失标记点")
print("  3. SimpleBlobDetector 在 DoG 结果上检测圆形 blob")
print("  4. 检测到的关键点是无序的，需要下一步用 PCA 排序")
print(f"  5. 当前检测到 {len(keypoints)} 个点，期望 336 个（16×21）")
print("\n✅ demo_02 完成！")
print("  下一步: demo_03_pca_sort.py — 学习 PCA 网格排序")