"""
SGBM 深度估计
=============
封装 cv2.StereoSGBM，接收立体校正后的左右图像对，
输出视差图和（可选）经 Q 矩阵重投影的三维点云。

Mock 模式：当 calibration_available=False 时，返回基于图像亮度反推的伪深度图，
仅用于在无相机环境下测试整体流程。
"""

from __future__ import annotations
import logging

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class SGBMDepthEstimator:
    """
    双目 SGBM 深度估计器。

    使用方式：
        estimator = SGBMDepthEstimator.from_config("config.yaml")
        disparity = estimator.compute_disparity(left_rect, right_rect)
        points_3d = estimator.disparity_to_3d(disparity, Q)  # Q 来自 StereoRectifier
    """

    def __init__(
        self,
        min_disparity: int = 0,
        num_disparities: int = 64,
        block_size: int = 7,
        uniqueness_ratio: int = 10,
        speckle_window_size: int = 100,
        speckle_range: int = 32,
        disp12_max_diff: int = 1,
        use_wls_filter: bool = False,  # WLS 后处理滤波（改善边缘质量，稍慢）
    ):
        bs = block_size  # 简写
        self._stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=bs,
            P1=8  * 3 * bs * bs,
            P2=32 * 3 * bs * bs,
            disp12MaxDiff=disp12_max_diff,
            uniquenessRatio=uniqueness_ratio,
            speckleWindowSize=speckle_window_size,
            speckleRange=speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self._use_wls = use_wls_filter
        if use_wls_filter:
            self._wls = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self._stereo)
            self._stereo_right = cv2.ximgproc.createRightMatcher(self._stereo)

        logger.info(
            "SGBMDepthEstimator 初始化，numDisp=%d blockSize=%d WLS=%s",
            num_disparities, block_size, use_wls_filter,
        )

    @classmethod
    def from_config(cls, config_path: str) -> "SGBMDepthEstimator":
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        s = cfg.get("sgbm", {})
        return cls(
            min_disparity       = s.get("min_disparity",        0),
            num_disparities     = s.get("num_disparities",      64),
            block_size          = s.get("block_size",           7),
            uniqueness_ratio    = s.get("uniqueness_ratio",     10),
            speckle_window_size = s.get("speckle_window_size",  100),
            speckle_range       = s.get("speckle_range",        32),
            disp12_max_diff     = s.get("disp12_max_diff",      1),
        )

    # ──────────────────────────────────────────────────────────────────────

    def compute_disparity(
        self, left_rect: np.ndarray, right_rect: np.ndarray
    ) -> np.ndarray:
        """
        计算视差图。

        Parameters
        ----------
        left_rect, right_rect : 立体校正后的灰度或 BGR 图像

        Returns
        -------
        disparity : float32 视差图，单位 px；无效点值为 0。
        """
        gray_l = self._to_gray(left_rect)
        gray_r = self._to_gray(right_rect)

        if self._use_wls:
            disp_l = self._stereo.compute(gray_l, gray_r)
            disp_r = self._stereo_right.compute(gray_r, gray_l)
            disp_filtered = self._wls.filter(disp_l, gray_l, disparity_map_right=disp_r)
            raw = disp_filtered.astype(np.float32) / 16.0
        else:
            raw = self._stereo.compute(gray_l, gray_r).astype(np.float32) / 16.0

        # 将无效（负值）视差置 0
        raw[raw < 0] = 0
        return raw

    @staticmethod
    def disparity_to_depth(
        disparity: np.ndarray,
        focal_length_px: float,
        baseline_mm: float,
    ) -> np.ndarray:
        """
        视差图→深度图（毫米）。

        Z = f * B / d，无效点（d≤0）设为 0。
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(
                disparity > 0,
                (focal_length_px * baseline_mm) / disparity,
                0.0,
            )
        return depth.astype(np.float32)

    @staticmethod
    def disparity_to_3d(
        disparity: np.ndarray, Q: np.ndarray
    ) -> np.ndarray:
        """
        使用 stereoRectify 输出的 Q 矩阵将视差图重投影为三维点云。

        Returns
        -------
        points_3d : (H, W, 3) float32，单位与 Q 一致（通常为毫米）
        """
        points = cv2.reprojectImageTo3D(disparity, Q)
        # 无效视差点设为 NaN
        mask_invalid = (disparity <= 0)
        points[mask_invalid] = np.nan
        return points.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def make_mock_disparity(
        image: np.ndarray,
        baseline_mm: float = 120.0,
        focal_px: float = 1066.0,
        work_z_mm: float = 800.0,
        noise_sigma: float = 2.0,
    ) -> np.ndarray:
        """
        Mock 深度图：根据图像亮度生成伪视差图，仅用于测试。
        绿色区域（芦笋）被赋予略高视差（对应较近距离，即较高 z）。
        """
        # 平均视差（对应工作距离）
        base_disp = (focal_px * baseline_mm) / work_z_mm

        gray = SGBMDepthEstimator._to_gray(image)
        # 绿色分量提取（BGR 格式：g=channel 1，b=0，r=2）
        if image.ndim == 3:
            green_excess = image[:, :, 1].astype(np.float32) - image[:, :, 2].astype(np.float32)
        else:
            green_excess = np.zeros_like(gray, dtype=np.float32)

        # 绿色区域视差更大（芦笋更近）
        disparity = np.full(gray.shape, base_disp, dtype=np.float32)
        disparity += np.clip(green_excess / 10.0, 0, 5)  # 绿色区域加 0~5px

        # 加噪
        rng = np.random.default_rng()
        disparity += rng.normal(0, noise_sigma, disparity.shape).astype(np.float32)
        disparity = np.clip(disparity, 1.0, None)
        return disparity

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
