<h1>五、Homography 单应性变换（透视矫正）</h1>

<h2>5.1 什么是 Homography？</h2>

<p>Homography（单应性变换）是一个 <b>3×3 的矩阵</b>，能把一个四边形变成另一个四边形。</p>

<callout emoji="💡" background-color="light-blue" border-color="blue">
  <p><b>一句话理解：</b>给 Homography 4 对对应点（源四边形 → 目标四边形），它就能算出一个变换矩阵，把整张图像从透视变形"掰正"成矩形。</p>
</callout>

<h2>5.2 数学原理</h2>

<p>Homography 矩阵 H 是 3×3 的：</p>

<pre lang="text"><code>    [ h11  h12  h13 ]
H = [ h21  h22  h23 ]
    [ h31  h32   1  ]</code></pre>

<ul>
  <li>右下角固定为 1，所以只有 <b>8 个未知数</b></li>
  <li>需要 <b>4 对对应点</b> 才能解出（4 个点 × 2 个坐标 = 8 个方程）</li>
  <li>使用齐次坐标进行计算</li>
</ul>

<h2>5.3 Homography 能做什么？</h2>

<table>
  <thead><tr><th>能力</th><th>说明</th></tr></thead>
  <tbody>
    <tr><td>透视矫正</td><td>处理近大远小的透视效应 ✅</td></tr>
    <tr><td>旋转</td><td>图像旋转任意角度 ✅</td></tr>
    <tr><td>缩放</td><td>整体放大缩小 ✅</td></tr>
    <tr><td>平移</td><td>整体移动 ✅</td></tr>
    <tr><td>镜头畸变</td><td>桶形/枕形畸变 ❌（非线性，处理不了）</td></tr>
    <tr><td>局部变形</td><td>硅胶按压的局部形变 ❌</td></tr>
  </tbody>
</table>

<h2>5.4 demo_04_homography.py 步骤拆解</h2>

<ol>
  <li><b>检测并排序标记点</b>（复用 demo_03 的 PCA + k-means + 贪心分配）</li>
  <li><b>提取四个角点</b>：从 grid_points 中取四个角 TL/TR/BR/BL</li>
  <li><b>定义目标矩形</b>：设定 pixel_spacing 和 margin，人工定义矫正后的标准矩形</li>
  <li><b>计算 H 矩阵 + 应用变换</b>：cv2.getPerspectiveTransform + cv2.warpPerspective</li>
  <li><b>分析矫正效果</b>：对比原始和矫正后的行/列间距均匀度</li>
  <li><b>可视化</b>：6 张子图对比</li>
</ol>

<h2>5.5 关键代码</h2>

<pre lang="python" caption="核心两行代码"><code>H = cv2.getPerspectiveTransform(corners_src, corners_dst)
warped = cv2.warpPerspective(img, H, (rect_w_int, rect_h_int))</code></pre>

<ul>
  <li><code>cv2.getPerspectiveTransform(src, dst)</code>：输入 4 对对应点，输出 3×3 的 H 矩阵</li>
  <li><code>cv2.warpPerspective(img, H, size)</code>：把整张图用 H 矩阵变换过去</li>
</ul>

<h2>5.6 效果评估：行/列间距对比</h2>

<table>
  <thead><tr><th>指标</th><th>原始图像</th><th>矫正后</th><th>改善</th></tr></thead>
  <tbody>
    <tr><td>行间距 mean</td><td>~17.5 px</td><td>~20.0 px</td><td>接近目标值</td></tr>
    <tr><td>行间距 std</td><td>~1.2 px</td><td>~0.3 px</td><td>减小 75%</td></tr>
    <tr><td>列间距 mean</td><td>~16.5 px</td><td>~20.0 px</td><td>接近目标值</td></tr>
    <tr><td>列间距 std</td><td>~1.5 px</td><td>~0.3 px</td><td>减小 80%</td></tr>
  </tbody>
</table>

<callout emoji="📊" background-color="light-green" border-color="green">
  <p><b>核心指标：std（标准差）</b></p>
  <p>std 越小说明间距越均匀，网格越规整。Homography 把 std 从 ~1.3 降到 ~0.3，效果非常显著。</p>
</callout>

<h2>5.7 残留误差分析</h2>

<p>即使经过 Homography 矫正，仍有少量误差（±1~2px），主要原因：</p>

<ul>
  <li><b>镜头畸变</b>：非线性畸变，Homography 是线性变换处理不了</li>
  <li><b>单点检测误差</b>：个别标记点附近有反光/噪声，导致定位偏差</li>
  <li><b>边缘点误差放大</b>：角点检测误差会被 Homography 放大到边缘区域</li>
</ul>

<callout emoji="❓" background-color="light-yellow" border-color="yellow">
  <p><b>实际案例：</b>第 4-5 行、第 14 列的行间距偏差达 -2.09 px，经排查是该点下方有白色反光，干扰了 SimpleBlobDetector 的检测，属于单点检测误差，不影响整体效果。</p>
</callout>

<hr/>

<h1>六、Warp 逐单元仿射变换（高精度矫正）</h1>

<h2>6.1 为什么需要 Warp？</h2>

<p>Homography 只有一个全局的 3×3 矩阵，是<b>整体线性变换</b>，处理不了非线性的镜头畸变和局部变形。</p>

<callout emoji="💡" background-color="light-blue" border-color="blue">
  <p><b>Warp 的思路：</b>把网格分成很多小格子（四边形 → 两个三角形），每个小格子单独算一个局部仿射变换。这样镜头畸变、边缘误差都能被逐个矫正。</p>
  <p>打个比方：Homography 是一张"大地图"统一缩放，Warp 是给每个小格子量身定制矫正方案。</p>
</callout>

<h2>6.2 核心算法：三角剖分 + 逐单元仿射</h2>

<h3>仿射变换 vs 透视变换</h3>

<table>
  <thead><tr><th>特征</th><th>仿射变换 (Affine)</th><th>透视变换 (Perspective)</th></tr></thead>
  <tbody>
    <tr><td>矩阵大小</td><td>2×3</td><td>3×3</td></tr>
    <tr><td>自由度</td><td>6</td><td>8</td></tr>
    <tr><td>需要对应点</td><td>3 对</td><td>4 对</td></tr>
    <tr><td>平行线</td><td>保持平行 ✅</td><td>不保持 ❌</td></tr>
    <tr><td>应用</td><td>每个小三角形</td><td>全局矫正</td></tr>
  </tbody>
</table>

<h3>三角剖分</h3>

<p>每个四边形单元沿对角线拆成两个三角形：</p>

<pre lang="text"><code>四边形 → 两个三角形
┌────┐
│ ↘  │    三角形1: TL-TR-BR
│    │    三角形2: TL-BR-BL
│ ↗  │
└────┘</code></pre>

<p>每个三角形用 3 对对应点算一个仿射变换矩阵，三角形内的所有像素都用这个矩阵映射。</p>

<h2>6.3 demo_05_warp.py 步骤拆解</h2>

<ol>
  <li><b>检测并排序网格点</b>：21×16 的有序网格</li>
  <li><b>计算原始网格间距</b>：取中位数作为参考间距</li>
  <li><b>边界外推（扩展一圈）</b>：在网格外围扩展一圈虚拟点（23×18），保证边缘完整</li>
  <li><b>生成标准网格</b>：按物理间距（0.6mm）生成均匀的理想网格</li>
  <li><b>逐三角形仿射映射</b>：每个三角形算仿射矩阵，生成 map_x/map_y 查找表</li>
  <li><b>保存 warp + 应用矫正</b>：用 cv2.remap 应用查找表</li>
  <li><b>可视化</b>：6 张子图对比</li>
</ol>

<h2>6.4 关键概念：查找表（map_x / map_y）</h2>

<p>Warp 的最终产物是两张和输出图像一样大的查找表：</p>

<ul>
  <li><code>map_x[y, x]</code>：矫正图像中 (x, y) 位置，对应原始图像的<b>列坐标</b></li>
  <li><code>map_y[y, x]</code>：矫正图像中 (x, y) 位置，对应原始图像的<b>行坐标</b></li>
</ul>

<p>这就是一张"导航图"——告诉矫正图像的每个像素："你原本应该在原始图像的哪里采样"。</p>

<p>应用时只需一行代码：</p>

<pre lang="python"><code>rectified = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)</code></pre>

<h2>6.5 边界外推问题分析</h2>

<p><b>现象：</b>矫正后图像左右边缘的标记点明显比中间大（被拉伸）。</p>

<p><b>原因：</b>原始图像的边界外推距离（0.5 倍间距，~8px）远小于标准网格的边距（PAD_PX=20px），导致原始图像中很窄的边界区域被映射到标准网格中较宽的区域 → 内容被拉伸。</p>

<p><b>数据验证（局部缩放率）：</b></p>

<table>
  <thead><tr><th>位置</th><th>gx (x方向缩放)</th><th>实际放大率 (1/gx)</th></tr></thead>
  <tbody>
    <tr><td>中间</td><td>0.849</td><td>1.18x</td></tr>
    <tr><td>左边缘</td><td>0.397</td><td>2.52x</td></tr>
    <tr><td>右边缘</td><td>0.397</td><td>2.52x</td></tr>
  </tbody>
</table>

<p><b>解决方法：</b>调整外推比例，从 0.5 倍间距改为 1.5 倍间距，让原始外推区域和标准边距更匹配。</p>

<h2>6.6 Warp vs Homography 对比</h2>

<table>
  <thead><tr><th>指标</th><th>Homography</th><th>Warp</th></tr></thead>
  <tbody>
    <tr><td>变换类型</td><td>全局线性</td><td>局部仿射</td></tr>
    <tr><td>矩阵数量</td><td>1 个 (3×3)</td><td>每格 2 个 (2×3)</td></tr>
    <tr><td>镜头畸变</td><td>❌ 不能处理</td><td>✅ 可以处理</td></tr>
    <tr><td>局部变形</td><td>❌ 不能处理</td><td>✅ 可以处理</td></tr>
    <tr><td>计算量</td><td>小</td><td>大（但只算一次）</td></tr>
    <tr><td>精度</td><td>中等</td><td>高</td></tr>
  </tbody>
</table>

<callout emoji="⭐" background-color="light-green" border-color="green">
  <p><b>结论：</b>Homography 是"粗矫正"，Warp 是"精矫正"。实际应用中，先做 Homography 快速预览，最终高精度重建用 Warp。</p>
</callout>
