# 芦笋分拣视觉系统：软件架构

> **本文档是软件实现的总体架构参考。**
> 与 [design_conclusion.md](design_conclusion.md) 共同构成项目设计基础。
> 详细检测算法设计见 [plans/2026-05-24-detection-design.md](plans/2026-05-24-detection-design.md)。

---

## 技术栈总览

| 层次 | 技术选型 | 理由 |
|:---|:---|:---|
| 编程语言 | Python 3.11+ | 生态丰富，OpenCV/NumPy 原生支持 |
| 深度估计 | OpenCV StereoSGBM | 无需 GPU，零数据，开箱即用 |
| 目标检测 | 传统 CV（HSV + 轮廓分析）| Phase 1；Phase 2 升级为 YOLOv8n-seg |
| 消息总线 | Python `queue.Queue` | 4 模块规模，无需 ROS2 |
| 硬件通信 | BLE（`bleak` 库） | 与下位机无线通信，发送拨钩指令 |
| 配置管理 | `PyYAML` + `config.yaml` | 相机参数、检测阈值集中管理 |
| 标定工具 | OpenCV `stereoCalibrate` | 标准双目标定流程 |

---

## 模块划分

```
sorter_mini_vision/
├── config.yaml                  # 全局配置（相机参数、检测阈值）
├── main.py                      # 主程序入口，启动4个线程
│
├── camera/
│   ├── capture.py               # 【采集线程】双目图像采集
│   └── rectify.py               # 立体校正（使用标定结果）
│
├── vision/
│   ├── sgbm.py                  # SGBM 视差计算
│   ├── detector.py              # 芦笋检测主逻辑（颜色→轮廓→位姿）
│   ├── depth_integrator.py      # 深度集成与层级判断
│   └── selector.py              # 目标选择（最右侧底层芦笋）
│
├── planner/
│   └── hook_planner.py          # 【规划线程】拨钩目标点计算
│
├── comm/
│   └── ble_client.py            # 【BLE通信线程】发送指令给下位机
│
├── calibration/                 # 标定工具（独立运行，非实时）
│   ├── stereo_calibrate.py      # 双目内外参标定
│   ├── extrinsic_calibrate.py   # 相机-世界外参标定
│   └── offset_calibrate.py      # d_cam_offset 标定
│
└── tools/                       # 开发调试工具
    ├── hsv_tuner.py             # HSV 阈值可视化调参工具 ← 第一个交付物
    ├── depth_viewer.py          # 深度图可视化
    └── replay.py                # 离线回放已保存的图像对
```

---

## 线程架构与数据流

```
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│  启动4个守护线程，监控状态，处理 Ctrl+C 退出              │
└─────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
  ┌─────────────┐                                          
  │ 采集线程    │                                          
  │ capture.py  │                                          
  │             │  queue_frames                            
  │ 双目相机    │ ──────────────►                          
  │ 图像对      │               │                          
  └─────────────┘               ▼                          
                        ┌──────────────┐                   
                        │  识别线程    │                   
                        │ detector.py  │                   
                        │              │  queue_snapshots  
                        │ SGBM + CV    │ ──────────────►  
                        │ → Snapshot   │               │   
                        └──────────────┘               ▼   
                                              ┌──────────────┐
                                              │  规划线程    │
                                              │ hook_planner │
                                              │              │  queue_commands
                                              │ → 拨钩坐标   │ ──────────────►
                                              └──────────────┘               │
                                                                              ▼
                                                                    ┌──────────────┐
                                                                    │  BLE通信线程 │
                                                                    │ ble_client   │
                                                                    │              │
                                                                    │ → 下位机     │
                                                                    └──────────────┘
```

**Queue 类型说明**：
- `queue_frames`：`Queue(maxsize=2)`，采集慢时丢弃旧帧
- `queue_snapshots`：`Queue(maxsize=5)`，缓冲多个识别结果
- `queue_commands`：`Queue(maxsize=10)`，等待下位机执行

---

## BLE 通信协议（草案）

下位机（蓝牙设备）接收以下指令包：

```python
# 指令格式（JSON over BLE Notify/Write）
{
    "cmd": "HOOK",           # 命令类型
    "y_mm": 145.3,           # 拨钩 Y 轴目标位置 [mm]
    "z_mm": 12.0,            # 拨钩 Z 轴插入高度 [mm]
    "seq": 42                # 序列号（用于确认）
}

# 下位机回复
{
    "ack": 42,               # 确认序列号
    "status": "DONE"         # 执行状态：DONE / ERROR
}
```

> **[!NOTE]**
> BLE 使用 `bleak` 库（异步，跨平台）。服务 UUID 和特征 UUID 需与下位机固件协商确定。

---

## 开发阶段规划

### Phase 0：工具优先（无硬件可完成）
- [ ] `tools/hsv_tuner.py` — HSV 调参工具（用测试图片）
- [ ] `tools/depth_viewer.py` — 深度图可视化
- [ ] `calibration/stereo_calibrate.py` — 双目标定脚本
- [ ] `config.yaml` 结构定义

### Phase 1：核心管线（需要相机）
- [ ] `camera/capture.py` — 采集线程
- [ ] `vision/detector.py` — 芦笋检测
- [ ] `vision/sgbm.py` — 深度估计
- [ ] `vision/depth_integrator.py` — 层级判断
- [ ] `main.py` — 单线程串行版本（先跑通，再改多线程）

### Phase 2：集成与通信（需要下位机）
- [ ] `comm/ble_client.py` — BLE 通信
- [ ] `planner/hook_planner.py` — 拨钩规划
- [ ] 多线程版本 `main.py`
- [ ] 端到端联调

### Phase 3：优化（可选）
- [ ] 替换检测模块为 YOLOv8n-seg（需要标注数据）
- [ ] WLS 滤波改善深度图质量
- [ ] 性能剖析与瓶颈优化

---

## 核心数据结构（来自 design_conclusion.md）

```python
@dataclass
class AsparagusPose:
    x_center: float      # 芦笋中心 X [mm]
    y_center: float      # 芦笋中心 Y [mm]
    z_top: float         # 顶面高度 [mm]，Z=0 为传送带平面
    angle_deg: float     # 与 X 轴夹角 [°]，范围 ±15°
    diameter_mm: float   # 直径 [mm]
    layer: int           # 0=底层，1=叠压层

@dataclass
class SceneSnapshot:
    timestamp_ms: int
    conveyor_position_mm: float
    asparagus_list: list[AsparagusPose]
    rightmost_target: AsparagusPose | None
```
