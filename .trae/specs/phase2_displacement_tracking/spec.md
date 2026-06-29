# Phase 2：位移计算 & 标记修复 - Product Requirement Document

## Overview
- **Summary**: 基于已完成的 Phase 1 几何矫正成果，完成 Phase 2 标记点跟踪的全部代码实现。包括：逐点位移向量计算、位移场热力图可视化、Inpaint 标记点修复、Warp 矫正后位移场分析，以及从 400 张接触图像中选取 10 张进行批量验证并可视化呈现。所有检测/排序/warp 逻辑必须复用 tac_utils.py 中 Phase 1 已验证的代码，确保项目连贯。
- **Purpose**: 在几何矫正的基础上，实现标记点位移的精确计算与可视化，为 Phase 3 光度立体法向估计提供无标记点干扰的矫正后图像，并建立位移场→形变→力估计的数据通路。
- **Target Users**: 学习者（用户本人），用于逐行调试学习、算法验证和后续飞书文档整理。

## Goals
- **G1**: 计算参考图像与接触图像之间每个标记点的精确位移向量 (dx, dy)
- **G2**: 在原始图像坐标下可视化位移场（箭头向量图、位移大小热力图）
- **G3**: 对参考图像和接触图像应用 Phase 1 构建的 Warp 进行几何矫正
- **G4**: 在矫正后图像上实现标记点 Inpaint 修复（去除黑色标记点对后续光度立体的干扰）
- **G5**: 在矫正后坐标下计算并可视化位移场（包含物理单位 mm）
- **G6**: 从 400 张接触图像中选取 10 张（含不同按压力度/位置的帧），批量验证算法正确性
- **G7**: 所有验证结果生成可视化图表，标注文件名，便于后续调试学习和飞书文档记录

## Non-Goals (Out of Scope)
- Phase 3 光度立体与 MLP 法向预测
- Phase 4 深度重建（Poisson/Dirichlet 积分）
- Phase 5 三维力估计（Hertz/FEM 接触模型）
- 实时视频流处理（单帧处理即可）
- 网络训练或模型推理

## Background & Context
- Phase 1 已完成并验证：DoG 标记点检测、PCA+k-means 网格排序、三角剖分稠密 Warp 构建
- 公共工具库 [tac_utils.py](file:///d:/02_Life-Long%20Learning/Project/04_Tactile_demo/tac_utils.py) 已封装所有 Phase 1 验证过的核心函数：`detect_and_sort()`、`build_warp()`、`apply_warp()`、`save_warp()/load_warp()`
- demo_07 已完成：参考图像与接触图像的检测+排序，原始坐标下的红绿点叠加对比和位移箭头图（2倍放大）
- 参考项目 viewtacv2 的关键实现：
  - `src/rectify/remap.py`: `inpaint_bgr()` 使用 cv2.INPAINT_TELEA 修复标记点
  - `src/rectify/gen_inpaint_mask.py`: 在矫正后图像上根据标准网格位置绘制圆形遮罩
  - `src/processing/tactile/depth_measure/preprocess.py`: `preprocess_raw_to_rectified_crop()` 完整预处理流程（inpaint → rectify → crop）
- 数据来源：`D:\02_Life-Long Learning\Project\01_Visual_tactile\tactile\data\data_wenli\`，包含 ref.jpg（参考）和 400 张 image_*.jpg（接触帧）
- 网格规格：16列 × 21行 = 336 个标记点，物理间距 0.6mm
- Warp 参数在 tac_utils.py 中已定义：OUT_WIDTH=320, PAD_PX=20, MASK_BOTTOM_PX=20

## Functional Requirements
- **FR-1**: 使用 tac_utils.detect_and_sort() 对参考图像和接触图像进行检测排序，输出 (21,16,2) 网格点坐标
- **FR-2**: 对每对 (ref, contact) 网格点计算位移向量场 disp = contact - ref，输出 (21,16,2) 数组，空单元格处为 NaN
- **FR-3**: 使用 tac_utils.build_warp() 从 ref.jpg 构建 Warp（map_x, map_y），或直接复用 Phase 1 已保存的 warp npz
- **FR-4**: 使用 tac_utils.apply_warp() 将原始图像矫正为标准网格图像，并裁剪到标记区域
- **FR-5**: 在矫正后图像上，根据已知的标准网格位置（均匀分布）生成圆形标记点遮罩，使用 cv2.inpaint (TELEA) 修复标记点
- **FR-6**: 在矫正后坐标下，由于网格是均匀的，直接用网格索引对齐参考和接触点，计算矫正后位移场（物理单位 mm）
- **FR-7**: 生成位移大小热力图（colormap），标注位移最大值、最小值和接触区域
- **FR-8**: 从 400 张接触图像中选取 10 张（按文件名均匀采样，覆盖按压过程的不同阶段）
- **FR-9**: 对选定的 10 张图像逐一运行完整流程，生成汇总可视化图表
- **FR-10**: 所有输出图片标注图像文件名、位移统计量（均值/最大/最小）

## Non-Functional Requirements
- **NFR-1**: 代码结构清晰，与 Phase 1 风格一致（demo_08/demo_09/... 逐步递进，每个 demo 专注一个知识点）
- **NFR-2**: 所有检测/排序/warp 参数必须直接从 tac_utils 导入，不允许在 demo 文件中重复定义
- **NFR-3**: 每个 demo 运行后在 outputs/ 目录生成可视化图片
- **NFR-4**: 中文注释，便于逐行调试学习
- **NFR-5**: Inpaint 修复后的图像不能有明显的黑色标记点残留，修复区域与周围颜色自然过渡

## Constraints
- **Technical**: Python 3.x, OpenCV, NumPy, Matplotlib（已有依赖，不需要额外安装）；不使用 PyTorch/深度学习
- **Business**: 基于 viewtacv2 项目但简化为教学 demo，核心算法保持一致但代码更易读
- **Dependencies**: 完全依赖 tac_utils.py 中的 Phase 1 成果；Warp 从 ref.jpg 一次构建，所有接触帧复用

## Assumptions
- 参考图像 ref.jpg 的检测结果为 336 点，0 空单元格（已验证）
- 接触图像在正常按压下应能检测到 ≥ 330 个点，极少数边缘点可能漏检
- 标记点直径在原始图像中约 6-8 像素，矫正后约 5-7 像素，Inpaint 半径取 3-5 像素
- Warp 一次构建后可以复用于所有 400 张接触图像（因为相机和传感器位置固定）

## Acceptance Criteria

### AC-1: Warp 构建与复用
- **Given**: ref.jpg 可用，tac_utils.py 中 build_warp 函数正确
- **When**: 运行 demo 构建 Warp 并保存
- **Then**: 成功生成 map_x, map_y 查找表，应用后输出矫正图像，网格均匀
- **Verification**: `programmatic`
- **Notes**: 可直接复用 tac_utils.build_warp()

### AC-2: Inpaint 标记修复效果
- **Given**: 矫正后图像，已知标准网格位置
- **When**: 生成标记点遮罩并调用 cv2.inpaint
- **Then**: 修复后图像中黑色标记点被自然填充，无明显黑斑或伪影
- **Verification**: `human-judgment`
- **Notes**: 对比修复前后图像，标记点区域应与周围凝胶颜色一致

### AC-3: 原始坐标下位移计算正确
- **Given**: ref 和 contact 的网格点均检测成功
- **When**: 计算 disp = contact - ref
- **Then**: 位移向量方向符合物理直觉（按压区域位移最大，远离区域位移趋近于 0），箭头可视化清晰
- **Verification**: `human-judgment`
- **Notes**: 接触区箭头应指向按压中心发散或汇聚方向

### AC-4: 矫正后位移场有物理单位
- **Given**: 矫正后 mm_per_px 已知（来自 warp meta）
- **When**: 将像素位移转换为 mm 位移
- **Then**: 最大位移在合理物理范围内（约 0.1~1mm，取决于按压力度）
- **Verification**: `programmatic`
- **Notes**: pitch 0.6mm / ~20px = ~0.03 mm/px，几像素位移约 0.1-0.2mm

### AC-5: 10 张图像批量验证无报错
- **Given**: 400 张接触图像
- **When**: 选取 10 张批量运行完整流程
- **Then**: 所有 10 张均成功处理，检测点数 ≥ 330，生成可视化结果
- **Verification**: `programmatic`
- **Notes**: 如某张图像检测点数 < 330，需记录但不中断流程

### AC-6: 汇总可视化图表清晰标注
- **Given**: 10 张图像处理完成
- **When**: 生成汇总图（2行5列或5行2列）
- **Then**: 每个子图标注文件名、位移统计量，热力图/向量图清晰可辨
- **Verification**: `human-judgment`

### AC-7: 代码复用 tac_utils 无重复
- **Given**: tac_utils.py 已封装 Phase 1 全部核心逻辑
- **When**: 检查 demo 代码
- **Then**: 检测/排序/warp/inpaint 相关核心算法均通过 tac_utils 调用，demo 中无重复的 DoG/PCA/k-means/Warp 构建代码
- **Verification**: `programmatic`

## Open Questions
- [ ] Inpaint 应该在原始图像做还是矫正后做？（viewtacv2 是在原始图像先 inpaint 再 rectify，这样更简单；但教学上可能需要两种都展示）
- [ ] 10 张验证图像的选取方式：均匀采样（每隔 40 张取一张）还是手动选取有代表性的帧？
