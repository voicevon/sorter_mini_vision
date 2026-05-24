"""
深度集成与层级判断
==================
将 AsparaguusDetector 的 2D 检测结果与 SGBM 视差图结合，
填充每根芦笋的 z_top 和 layer 字段。

同时负责将像素坐标（已有 XY）修正为完整的世界坐标 Z 分量。
"""

from __future__ import annotations
import logging

import numpy as np
import yaml

from models import AsparagusPose

logger = logging.getLogger(__name__)


class DepthIntegrator:
    """
    深度集成器：将视差图中的深度值采样到每根芦笋轮廓区域，
    计算 z_top（相对传送带平面的高度）并判断叠层。

    Parameters
    ----------
    focal_length_px : float
        左目相机焦距（像素）
    baseline_mm : float
        双目基线（毫米）
    work_distance_mm : float
        相机到传送带平面的工作距离（毫米）
    z_layer_threshold_mm : float
        层级判断阈值：芦笋顶面高于此值视为叠压层（layer=1）
    min_valid_pixels : int
        采样区域内有效视差点的最少数量；低于此值标记 depth_valid=False
    """

    def __init__(
        self,
        focal_length_px: float = 1066.0,
        baseline_mm: float = 120.0,
        work_distance_mm: float = 800.0,
        z_layer_threshold_mm: float = 5.0,
        min_valid_pixels: int = 10,
    ):
        self._focal  = focal_length_px
        self._B      = baseline_mm
        self._Z_work = work_distance_mm
        self._z_thresh = z_layer_threshold_mm
        self._min_px   = min_valid_pixels

        logger.info(
            "DepthIntegrator 初始化 f=%.1fpx B=%.1fmm Z_work=%.1fmm z_thresh=%.1fmm",
            focal_length_px, baseline_mm, work_distance_mm, z_layer_threshold_mm,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "DepthIntegrator":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cam = cfg.get("camera", {})
        det = cfg.get("detection", {})

        K_list = cam.get("K_left", [])
        if K_list and len(K_list) == 9:
            K = np.array(K_list).reshape(3, 3)
            fx = float(K[0, 0])
        else:
            w = cam.get("image_width", 1280)
            fx = w * (4.0 / 5.376)

        return cls(
            focal_length_px      = fx,
            baseline_mm          = cam.get("baseline_mm", 120.0),
            work_distance_mm     = cam.get("work_distance_mm", 800.0),
            z_layer_threshold_mm = det.get("z_layer_threshold_mm", 5.0),
        )

    # ──────────────────────────────────────────────────────────────────────
    # 主接口
    # ──────────────────────────────────────────────────────────────────────

    def integrate(
        self,
        poses: list[AsparagusPose],
        disparity_map: np.ndarray,
        contour_masks: list[np.ndarray],
    ) -> list[AsparagusPose]:
        """
        在每根芦笋的轮廓掩码区域内采样视差，计算 z_top 和 layer，
        就地修改 poses 并返回。

        Parameters
        ----------
        poses : detector.detect() 返回的列表（z_top/layer 待填充）
        disparity_map : SGBM 输出的 float32 视差图（单位 px）
        contour_masks : 与 poses 一一对应的二值掩码列表

        Returns
        -------
        poses : 填充完 z_top/layer/depth_valid 的同一列表
        """
        if len(poses) != len(contour_masks):
            logger.error(
                "poses(%d) 与 contour_masks(%d) 数量不匹配，跳过深度集成",
                len(poses), len(contour_masks),
            )
            return poses

        for asp, mask in zip(poses, contour_masks):
            # 在轮廓区域内提取有效视差（>0）
            disp_vals = disparity_map[mask > 0]
            valid = disp_vals[disp_vals > 0]

            if len(valid) < self._min_px:
                # 有效点太少，深度缺失
                asp.z_top       = 0.0
                asp.layer       = 0
                asp.depth_valid = False
                logger.debug("深度缺失（有效点 %d < %d），跳过", len(valid), self._min_px)
                continue

            # 取中值视差（鲁棒于边缘噪声）
            disp_median = float(np.median(valid))

            # 视差 → 相机深度（Z 轴距离）
            z_camera = (self._focal * self._B) / disp_median  # mm

            # 相对传送带平面高度（向上为正）
            z_top = self._Z_work - z_camera  # 相机近 → z 更大 → z_top 更大
            z_top = round(max(0.0, z_top), 2)  # 不允许负值（芦笋不在传送带下方）

            asp.z_top       = z_top
            asp.layer       = 1 if z_top > self._z_thresh else 0
            asp.depth_valid = True

        logger.debug(
            "[深度] 填充完成，layer1=%d / total=%d",
            sum(1 for a in poses if a.layer == 1),
            len(poses),
        )
        return poses

    def integrate_mock(self, poses: list[AsparagusPose]) -> list[AsparagusPose]:
        """
        Mock 深度集成：不依赖真实视差图，
        根据检测顺序随机分配 z_top（用于流程测试）。
        """
        rng = np.random.default_rng()
        for i, asp in enumerate(poses):
            # 模拟：部分芦笋叠压（z_top > 阈值）
            is_stacked = (i % 3 == 2)  # 每 3 根有 1 根叠层
            asp.z_top       = float(rng.uniform(8, 15)) if is_stacked else float(rng.uniform(0, 3))
            asp.layer       = 1 if asp.z_top > self._z_thresh else 0
            asp.depth_valid = True
        return poses
