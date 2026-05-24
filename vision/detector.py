"""
芦笋检测器
==========
输入：去畸变后的左目 BGR 图像
输出：检测到的芦笋列表（含 2D 位姿），深度和层级由 depth_integrator 补充。

处理流程：
    BGR → HSV → 颜色阈值掩码 → 形态学清理 → 连通域分析
    → minAreaRect → 世界坐标 XY → (部分填充的) AsparagusPose 列表

世界坐标转换：
    本模块只负责 XY 坐标（从像素坐标反投影到 Z=Z_work 平面），
    z_top 和 layer 由 DepthIntegrator 填充。
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

import cv2
import numpy as np
import yaml

from models import AsparagusPose

logger = logging.getLogger(__name__)


@dataclass
class _RawDetection:
    """单根芦笋的原始检测结果（像素坐标 + 轮廓信息）。"""
    center_px: tuple[float, float]  # (u, v) 像素坐标
    angle_deg: float                 # 与水平轴夹角（已归一化到 ±90°）
    long_axis_px: float              # 长轴像素长度
    short_axis_px: float             # 短轴像素长度（≈直径）
    contour_mask: np.ndarray         # 单根芦笋的二值掩码（全图尺寸）
    area_px: float                   # 轮廓面积（像素）


class AsparaguusDetector:
    """
    芦笋 2D 检测器：HSV 分割 + 轮廓分析。

    Parameters
    ----------
    config_path : str
        config.yaml 路径
    """

    def __init__(
        self,
        hsv_lower: tuple[int, int, int] = (35, 40, 40),
        hsv_upper: tuple[int, int, int] = (85, 255, 255),
        morph_kernel_size: int = 5,
        min_area_px: int = 200,
        work_distance_mm: float = 800.0,
        focal_length_px: float = 1066.0,
        principal_point: tuple[float, float] | None = None,
    ):
        self._lower = np.array(hsv_lower, dtype=np.uint8)
        self._upper = np.array(hsv_upper, dtype=np.uint8)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self._min_area = min_area_px
        self._work_z = work_distance_mm
        self._focal  = focal_length_px
        self._cx = principal_point[0] if principal_point else None
        self._cy = principal_point[1] if principal_point else None

        # 像素→毫米转换比（在工作平面上）：mm/px = Z / f
        self._px_to_mm = work_distance_mm / focal_length_px

        logger.info(
            "AsparaguusDetector 初始化 HSV[%s~%s] min_area=%dpx px2mm=%.3f",
            hsv_lower, hsv_upper, min_area_px, self._px_to_mm,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "AsparaguusDetector":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        det = cfg.get("detection", {})
        cam = cfg.get("camera", {})

        lo = det.get("hsv_lower", [35, 40, 40])
        hi = det.get("hsv_upper", [85, 255, 255])
        z  = cam.get("work_distance_mm", 800.0)

        # 从内参矩阵提取焦距和主点（如果已标定）
        K_list = cam.get("K_left", [])
        if K_list and len(K_list) == 9:
            K = np.array(K_list).reshape(3, 3)
            fx = float(K[0, 0])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
        else:
            # 未标定时用估算值
            w = cam.get("image_width", 1280)
            h = cam.get("image_height", 720)
            fx = w * (4.0 / 5.376)  # 4mm 焦距 / 1/3'' 传感器宽度
            cx, cy = w / 2.0, h / 2.0

        return cls(
            hsv_lower        = tuple(lo),
            hsv_upper        = tuple(hi),
            morph_kernel_size= det.get("morph_kernel_size", 5),
            min_area_px      = det.get("min_area_px", 200),
            work_distance_mm = z,
            focal_length_px  = fx,
            principal_point  = (cx, cy),
        )

    # ──────────────────────────────────────────────────────────────────────
    # 主接口
    # ──────────────────────────────────────────────────────────────────────

    def detect(
        self,
        img_bgr: np.ndarray,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> list[AsparagusPose]:
        """
        在输入图像中检测芦笋，返回部分填充的 AsparagusPose 列表。
        z_top 和 layer 字段默认为 0，由 DepthIntegrator 后续填充。

        Parameters
        ----------
        img_bgr : 去畸变后的左目 BGR 图像
        image_width, image_height : 图像尺寸（用于设置主点默认值）
        """
        h, w = img_bgr.shape[:2]
        cx = self._cx if self._cx is not None else w / 2.0
        cy = self._cy if self._cy is not None else h / 2.0

        # 1. 颜色分割
        mask = self._segment_green(img_bgr)

        # 2. 连通域分析 + 轮廓提取
        raw_detections = self._extract_contours(mask, h, w)

        if not raw_detections:
            return []

        # 3. 转换到世界坐标 XY
        poses: list[AsparagusPose] = []
        for rd in raw_detections:
            u, v = rd.center_px
            # 像素→相机坐标（Z=工作距离）
            x_cam = (u - cx) * self._px_to_mm
            y_cam = (v - cy) * self._px_to_mm

            # 直径估算：短轴像素 × 像素尺寸
            diameter_mm = rd.short_axis_px * self._px_to_mm

            # 角度归一化到 ±15°（超出范围说明不是茎杆方向）
            angle = self._normalize_angle(rd.angle_deg)

            poses.append(AsparagusPose(
                x_center     = round(x_cam, 1),
                y_center     = round(y_cam, 1),
                z_top        = 0.0,    # 待 DepthIntegrator 填充
                angle_deg    = round(angle, 1),
                diameter_mm  = round(max(2.0, diameter_mm), 1),
                layer        = 0,      # 待 DepthIntegrator 填充
                contour_area_px = rd.area_px,
                depth_valid  = False,  # 待 DepthIntegrator 设置
            ))

        logger.debug("[检测] 识别到 %d 根芦笋", len(poses))
        return poses

    def get_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        """返回绿色掩码（调试/可视化用）。"""
        return self._segment_green(img_bgr)

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _segment_green(self, img_bgr: np.ndarray) -> np.ndarray:
        """HSV 颜色阈值 + 形态学清理，返回二值掩码。"""
        hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        return mask

    def _extract_contours(
        self, mask: np.ndarray, img_h: int, img_w: int
    ) -> list[_RawDetection]:
        """从掩码中提取各连通域的轮廓，返回原始检测结果列表。"""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        results: list[_RawDetection] = []

        for i in range(1, num_labels):  # 0 是背景
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self._min_area:
                continue

            # 提取单个连通域掩码
            single = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            contour = max(contours, key=cv2.contourArea)
            if len(contour) < 5:  # minAreaRect 需要至少 5 点
                continue

            rect = cv2.minAreaRect(contour)
            center, (w_px, h_px), angle = rect

            long_axis  = max(w_px, h_px)
            short_axis = min(w_px, h_px)

            # 长轴太短（碎片）或比例不像茎杆（长宽比 < 3）则跳过
            if long_axis < 30 or (short_axis > 0 and long_axis / short_axis < 3):
                continue

            # 角度对应长轴方向
            if w_px < h_px:
                angle += 90  # 使 angle 代表长轴方向

            results.append(_RawDetection(
                center_px    = (float(center[0]), float(center[1])),
                angle_deg    = float(angle),
                long_axis_px = float(long_axis),
                short_axis_px= float(short_axis),
                contour_mask = single,
                area_px      = float(area),
            ))

        return results

    @staticmethod
    def _normalize_angle(angle_deg: float) -> float:
        """
        将任意角度归一化到 ±90° 范围。
        芦笋大致水平，理论角度 ≈ 0°±15°。
        """
        a = angle_deg % 180
        if a > 90:
            a -= 180
        return a
