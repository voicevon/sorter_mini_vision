"""
拨钩规划模块
============
接收 SceneSnapshot，计算拨钩的 (y_mm, z_mm) 目标点，
输出 HookCommand 给 BLE 通信模块。

坐标变换链（来自 design_conclusion.md §四）：
    AsparagusPose (x, y, z) [世界坐标 mm]
        ↓ [固定偏置补偿 d_cam_offset，一次性标定]
    拨钩目标点 (Y_hook, Z_hook) [mm，世界坐标]
        ↓
    HookCommand → BLE 发送给下位机
"""

from __future__ import annotations
import logging

import yaml

from models import AsparagusPose, SceneSnapshot, HookCommand

logger = logging.getLogger(__name__)


class HookPlanner:
    """
    拨钩轨迹规划器。

    Parameters
    ----------
    d_cam_offset_mm : float
        相机Y轴坐标系与拨钩Y轴坐标系的偏置（标定后填入）
        Y_hook = Y_cam_measure + d_cam_offset
    hook_z_clearance_mm : float
        插入点 Z 高度 = z_top - z_clearance（拨钩从芦笋顶面以上插入）
    hook_min_z_mm : float
        拨钩最低插入高度（保护传送带表面）
    """

    def __init__(
        self,
        d_cam_offset_mm: float = 0.0,
        hook_z_clearance_mm: float = 5.0,
        hook_min_z_mm: float = 2.0,
    ):
        self._offset    = d_cam_offset_mm
        self._clearance = hook_z_clearance_mm
        self._min_z     = hook_min_z_mm
        self._seq       = 0

        logger.info(
            "HookPlanner 初始化 d_cam_offset=%.1fmm z_clearance=%.1fmm min_z=%.1fmm",
            d_cam_offset_mm, hook_z_clearance_mm, hook_min_z_mm,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "HookPlanner":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # 规划参数目前在 config 中未单独分节，使用默认值
        # TODO: 标定后将 d_cam_offset 写入 config.yaml planner 节
        planner_cfg = cfg.get("planner", {})
        return cls(
            d_cam_offset_mm    = planner_cfg.get("d_cam_offset_mm",    0.0),
            hook_z_clearance_mm= planner_cfg.get("hook_z_clearance_mm", 5.0),
            hook_min_z_mm      = planner_cfg.get("hook_min_z_mm",       2.0),
        )

    # ──────────────────────────────────────────────────────────────────────

    def plan(self, snapshot: SceneSnapshot) -> HookCommand | None:
        """
        根据 SceneSnapshot 计算拨钩指令。

        Returns
        -------
        HookCommand，若快照中没有合适目标则返回 None。
        """
        target: AsparagusPose | None = snapshot.rightmost_target

        if target is None:
            logger.info("[规划] 无目标，跳过本帧")
            return None

        # Y 轴目标：目标 Y + 相机偏置补偿
        y_hook = target.y_center + self._offset

        # Z 轴插入高度：从芦笋顶面以上 clearance 距离处插入
        z_hook = target.z_top - self._clearance
        z_hook = max(z_hook, self._min_z)  # 不低于最低保护高度

        self._seq += 1
        cmd = HookCommand(
            seq                = self._seq,
            y_mm               = round(y_hook, 2),
            z_mm               = round(z_hook, 2),
            source_snapshot_ts = snapshot.timestamp_ms,
        )

        logger.info(
            "[规划] 指令 #%d → y=%.1fmm z=%.1fmm (目标z_top=%.1fmm layer=%d)",
            cmd.seq, cmd.y_mm, cmd.z_mm, target.z_top, target.layer,
        )
        return cmd
