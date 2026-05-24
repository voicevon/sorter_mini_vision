# 目标检测方案设计：传统 CV 管线（Phase 1）

> **文档状态**：已批准，可进入实现阶段。
> **决策日期**：2026-05-24
> **关联文档**：[design_conclusion.md](../design_conclusion.md)、[architecture.md](../architecture.md)

---

## 背景与约束

| 约束项 | 决策 |
|:---|:---|
| 运行平台 | 笔记本电脑（x86，无独立 GPU） |
| 编程语言 | Python |
| 训练数据 | 无（零数据启动） |
| 实时性要求 | 无严格要求，先跑通再优化 |
| 芦笋排列 | 大致平行于 X 轴，±15° 偏角，可能 1~2 层叠压 |

**选型结论**：Phase 1 使用纯传统 CV（颜色分割 + 轮廓分析 + SGBM 深度集成），无需任何训练数据即可启动。Phase 2 可在积累 200~300 张标注图后升级为 YOLOv8n-seg，接口不变。

---

## 处理管线详细设计

### 模块 1：图像预处理

**输入**：双目相机左目 BGR 图像（未畸变校正）

**处理步骤**：

```python
# 1. 去畸变（使用标定后的内参矩阵 K 和畸变系数 D）
img_undist = cv2.undistort(img_raw, K_left, D_left)

# 2. 转换到 HSV 颜色空间
img_hsv = cv2.cvtColor(img_undist, cv2.COLOR_BGR2HSV)

# 3. 绿色阈值分割（参数写入 config.yaml，支持实时调节）
lower_green = np.array([H_min, S_min, V_min])  # 默认 [35, 40, 40]
upper_green = np.array([H_max, S_max, V_max])  # 默认 [85, 255, 255]
mask_raw = cv2.inRange(img_hsv, lower_green, upper_green)

# 4. 形态学操作（去噪 + 填洞）
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_clean = cv2.morphologyEx(mask_raw, cv2.MORPH_OPEN, kernel)   # 去噪
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel) # 填洞
```

**输出**：`mask_green`（二值图，255=芦笋区域）

**注意**：HSV 阈值是最关键的调节参数，需要配套一个可视化调参工具（Trackbar 小程序），在真实场景下标定一次后固化到 `config.yaml`。

---

### 模块 2：个体分割与 2D 位姿提取

**输入**：`mask_green`

**处理步骤**：

```python
# 1. 连通域分析，过滤噪声小区域
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_clean)

asparagus_contours = []
for i in range(1, num_labels):  # 0 是背景
    area = stats[i, cv2.CC_STAT_AREA]
    if area < MIN_AREA_PX:  # 默认 200px，过滤碎片
        continue
    # 提取单个连通域的掩码，找轮廓
    single_mask = (labels == i).astype(np.uint8) * 255
    contours, _ = cv2.findContours(single_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    asparagus_contours.append(contours[0])

# 2. 对每个轮廓做最小外接矩形，提取位姿
for contour in asparagus_contours:
    rect = cv2.minAreaRect(contour)
    center_px, (w_px, h_px), angle_deg = rect

    # 长轴方向即茎杆方向，角度归一化到 [-15, +15]°
    long_axis_px = max(w_px, h_px)
    short_axis_px = min(w_px, h_px)

    # 角度规范化：minAreaRect 返回 [-90, 0)，需转换
    if w_px < h_px:
        angle_deg += 90  # 使角度对应长轴方向

    # 直径估算（像素→毫米）
    diameter_mm = short_axis_px * PX_TO_MM  # PX_TO_MM 由相机内参和工作距离确定

    # 像素坐标反投影到世界坐标（X, Y）
    x_world, y_world = pixel_to_world_xy(center_px, Z_WORK, K_left, T_cam2world)
```

**输出**：每根芦笋的 `(center_px, x_world, y_world, angle_deg, diameter_mm, contour_mask)`

**角度约定**：
- `angle_deg = 0`：芦笋完全平行于 X 轴（传送带方向）
- `angle_deg > 0`：头部偏向 Y+（右偏）
- 范围锁定为 `[-15°, +15°]`，超出范围视为检测异常

---

### 模块 3：SGBM 深度集成与层级判断

**输入**：双目图像对（左图 + 右图，已完成立体校正）

**SGBM 参数配置**：

```python
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,          # 必须是 16 的倍数；工作距离 800mm 视差约 40px
    blockSize=7,                # 奇数，芦笋细时用小值
    P1=8 * 3 * 7**2,            # 平滑性惩罚（小变化）
    P2=32 * 3 * 7**2,           # 平滑性惩罚（大变化）
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
disparity = stereo.compute(img_left_rect, img_right_rect).astype(np.float32) / 16.0
```

**深度集成**：

```python
for asp in asparagus_list:
    # 在芦笋掩码区域内提取视差，取中值（鲁棒于噪声）
    disp_roi = disparity[asp.contour_mask > 0]
    disp_valid = disp_roi[disp_roi > 0]  # 过滤无效视差（0 值）

    if len(disp_valid) < 10:  # 有效点太少，跳过
        asp.z_top = None
        continue

    disp_median = np.median(disp_valid)
    # 视差→深度→Z 世界坐标
    z_camera = (focal_length * baseline) / disp_median
    asp.z_top = z_camera - Z_WORK  # 相对于传送带平面的高度
```

**层级判断规则**：

```python
Z_LAYER_THRESHOLD = 5.0  # mm，高于此值视为叠压层

for asp in asparagus_list:
    if asp.z_top is None:
        asp.layer = 0  # 深度缺失，保守处理
    elif asp.z_top > Z_LAYER_THRESHOLD:
        asp.layer = 1  # 叠压在其他芦笋之上
    else:
        asp.layer = 0  # 直接接触传送带
```

**输出**：完整的 `AsparagusPose` 列表（包含 `z_top` 和 `layer`）

---

### 模块 4：目标选择（最右侧待拨出目标）

```python
def select_rightmost_target(asparagus_list: list[AsparagusPose]) -> AsparagusPose | None:
    """
    选取最右侧（Y 最大）的底层芦笋作为拨钩目标。
    策略：优先拨底层（layer=0），避免叠层倒塌。
    """
    bottom_layer = [asp for asp in asparagus_list if asp.layer == 0]

    if not bottom_layer:
        # 退化情况：全部为叠层，取 z 最高者（最顶层）
        return max(asparagus_list, key=lambda a: a.z_top or 0) if asparagus_list else None

    return max(bottom_layer, key=lambda a: a.y_center)
```

---

## 关键配置参数（config.yaml）

```yaml
camera:
  work_distance_mm: 800       # 相机到传送带的工作距离（待标定确认）
  baseline_mm: 120            # 双目基线（待测量）
  K_left: []                  # 内参矩阵（标定后填入）
  D_left: []                  # 畸变系数（标定后填入）
  T_cam2world: []             # 相机到世界坐标系外参（标定后填入）

detection:
  hsv_lower: [35, 40, 40]    # HSV 绿色下界（需现场调节）
  hsv_upper: [85, 255, 255]  # HSV 绿色上界（需现场调节）
  min_area_px: 200            # 最小连通域面积（过滤碎片）
  z_layer_threshold_mm: 5.0  # 层级判断阈值

sgbm:
  num_disparities: 64
  block_size: 7
  uniqueness_ratio: 10
```

---

## 调参工具（必须先写）

在写主程序之前，**第一个交付物**应该是 HSV 调参工具：

```python
# tools/hsv_tuner.py
# 加载一张真实场景图片，用 Trackbar 实时调整 HSV 阈值
# 找到合适值后输出到控制台，写入 config.yaml
```

---

## Phase 2 升级路径

当积累 200+ 张标注图像后，可将模块 2 替换为：

```
YOLOv8n-seg（实例分割）
  → 输出每根芦笋的分割掩码 + 边界框
  → 后续深度集成和层级判断逻辑完全不变
```

接口兼容，升级无需改动模块 3 和模块 4。

---

## 风险与缓解

| 风险 | 可能性 | 缓解策略 |
|:---|:---|:---|
| 传送带绿色背景干扰分割 | 中 | 调整 V 通道下界；考虑改用黑色传送带 |
| 芦笋紧贴时分割成一个 blob | 高 | 对过宽轮廓做骨架分析（skeletonize）分离 |
| SGBM 在细芦笋上视差缺失 | 中 | 用 WLS 滤波补全；或退化为用颜色重心的 z |
| 叠层判断 z 阈值不准 | 低 | 现场用已知高度垫块验证，调整阈值 |
