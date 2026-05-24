"""
共享数据结构
============
视觉系统所有模块共用的数据类定义。
以 design_conclusion.md § 三 为准。
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class AsparagusPose:
    """单根芦笋的三维位姿描述（世界坐标系）。"""
    x_center: float       # 芦笋中心 X 坐标 [mm]，沿传送带方向
    y_center: float       # 芦笋中心 Y 坐标 [mm]，垂直传送带方向（拨钩运动方向）
    z_top: float          # 芦笋顶面高度 [mm]，Z=0 为传送带平面
    angle_deg: float      # 芦笋长轴与 X 轴的夹角 [°]，范围 ±15°
    diameter_mm: float    # 直径估计 [mm]，范围 4~40 mm
    layer: int            # 叠放层级：0=最底层，1=压在其他芦笋之上

    # 内部使用，不参与规划
    contour_area_px: float = 0.0   # 轮廓面积（像素），调试用
    depth_valid: bool = True        # 深度采样是否有效


@dataclass
class SceneSnapshot:
    """单次拍照识别结果，视觉模块输出给规划模块的完整数据包。"""
    timestamp_ms: int                          # 拍照时刻（系统时间，毫秒）
    conveyor_position_mm: float                # 拍照时传送带绝对位置 [mm]（编码器提供）
    asparagus_list: list[AsparagusPose] = field(default_factory=list)
    rightmost_target: AsparagusPose | None = None  # 最右侧待拨出目标

    @property
    def count(self) -> int:
        return len(self.asparagus_list)

    @property
    def bottom_layer_targets(self) -> list[AsparagusPose]:
        """返回所有底层（layer=0）芦笋，按 y_center 降序排列。"""
        return sorted(
            [a for a in self.asparagus_list if a.layer == 0],
            key=lambda a: a.y_center,
            reverse=True,
        )


@dataclass
class HookCommand:
    """规划模块输出给 BLE 通信模块的拨钩指令。"""
    seq: int               # 序列号（用于 ACK 确认）
    y_mm: float            # 拨钩 Y 轴目标位置 [mm]，世界坐标系
    z_mm: float            # 拨钩 Z 轴插入高度 [mm]，世界坐标系
    source_snapshot_ts: int = 0  # 对应 SceneSnapshot 的时间戳

    def to_dict(self) -> dict:
        return {
            "cmd": "HOOK",
            "seq": self.seq,
            "y_mm": round(self.y_mm, 2),
            "z_mm": round(self.z_mm, 2),
        }
