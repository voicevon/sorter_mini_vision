"""
目标选择器
==========
从 AsparagusPose 列表中选出当前应拨出的目标芦笋。

选取策略（来自 design_conclusion.md）：
    1. 优先选底层（layer=0）芦笋，避免叠层倒塌。
    2. 在底层中选 y_center 最大者（最右侧，最先进入拨钩工作区）。
    3. 退化情况（全为叠层）：选 z_top 最高者（最顶层，最好拨出）。
"""

from __future__ import annotations
import logging

from models import AsparagusPose, SceneSnapshot

logger = logging.getLogger(__name__)


def select_target(asparagus_list: list[AsparagusPose]) -> AsparagusPose | None:
    """
    从芦笋列表中选出当前拨钩目标。

    Parameters
    ----------
    asparagus_list : 完整填充的 AsparagusPose 列表

    Returns
    -------
    目标芦笋，若列表为空则返回 None。
    """
    if not asparagus_list:
        logger.debug("[选目标] 列表为空，无目标")
        return None

    # 1. 优先底层
    bottom = [a for a in asparagus_list if a.layer == 0]

    if bottom:
        target = max(bottom, key=lambda a: a.y_center)
        logger.debug(
            "[选目标] 底层目标 y=%.1fmm z=%.1fmm angle=%.1f°",
            target.y_center, target.z_top, target.angle_deg,
        )
        return target

    # 2. 退化：全为叠层，选最顶层
    target = max(asparagus_list, key=lambda a: a.z_top)
    logger.warning(
        "[选目标] 全为叠层！退化选 z_top 最高者 z=%.1fmm", target.z_top
    )
    return target


def build_snapshot(
    timestamp_ms: int,
    conveyor_position_mm: float,
    asparagus_list: list[AsparagusPose],
) -> SceneSnapshot:
    """
    构建完整的 SceneSnapshot。

    Parameters
    ----------
    timestamp_ms : 拍照时刻（毫秒）
    conveyor_position_mm : 传送带绝对位置（编码器读数，mm）
    asparagus_list : 完整填充的 AsparagusPose 列表
    """
    target = select_target(asparagus_list)
    snap = SceneSnapshot(
        timestamp_ms        = timestamp_ms,
        conveyor_position_mm= conveyor_position_mm,
        asparagus_list      = asparagus_list,
        rightmost_target    = target,
    )
    logger.info(
        "[快照] ts=%d 传送带=%.1fmm 芦笋数=%d 底层=%d 目标=%s",
        timestamp_ms,
        conveyor_position_mm,
        snap.count,
        len(snap.bottom_layer_targets),
        f"y={target.y_center:.1f}mm" if target else "无",
    )
    return snap
