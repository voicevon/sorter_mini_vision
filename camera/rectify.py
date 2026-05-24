"""
立体图像校正
============
使用 stereoCalibrate 输出的 R/T 矩阵，对左右图像对进行极线校正，
使左右图像的对应点在同一水平扫描线上，为 SGBM 视差计算做准备。

无标定数据时（calibration_available=False），直接透传原始图像（Mock 模式）。
"""

from __future__ import annotations
import logging
import numpy as np
import cv2
import yaml

logger = logging.getLogger(__name__)


class StereoRectifier:
    """
    双目图像立体校正器。

    使用方式：
        rectifier = StereoRectifier.from_config("config.yaml")
        left_rect, right_rect = rectifier.rectify(left_raw, right_raw)
    """

    def __init__(
        self,
        K_left: np.ndarray,
        D_left: np.ndarray,
        K_right: np.ndarray,
        D_right: np.ndarray,
        R: np.ndarray,
        T: np.ndarray,
        image_size: tuple[int, int],
    ):
        """
        Parameters
        ----------
        K_left, K_right : 3×3 内参矩阵
        D_left, D_right : 畸变系数向量
        R : 3×3 旋转矩阵（右相机相对左相机）
        T : 3×1 平移向量 [mm]
        image_size : (width, height)
        """
        self.image_size = image_size
        self.calibration_available = True

        # 立体校正计算
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_left, D_left, K_right, D_right,
            image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,  # alpha=0: 只保留有效像素，不留黑边
        )
        self.Q = Q  # 视差→深度的 4×4 重投影矩阵，供 depth_integrator 使用

        # 预计算校正映射（提升运行时性能）
        self._map1_left, self._map2_left = cv2.initUndistortRectifyMap(
            K_left, D_left, R1, P1, image_size, cv2.CV_16SC2)
        self._map1_right, self._map2_right = cv2.initUndistortRectifyMap(
            K_right, D_right, R2, P2, image_size, cv2.CV_16SC2)

        logger.info("StereoRectifier 初始化完成，图像尺寸 %s", image_size)

    @classmethod
    def from_config(cls, config_path: str) -> "StereoRectifier":
        """从 config.yaml 加载标定参数，构造实例。标定参数缺失时返回 Mock 实例。"""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cam = cfg.get("camera", {})
        K_left_list  = cam.get("K_left",  [])
        K_right_list = cam.get("K_right", [])

        if not K_left_list or not K_right_list:
            logger.warning("config.yaml 中缺少内参矩阵，使用 Mock 模式（直接透传图像）")
            return cls._make_mock()

        w = cam.get("image_width", 1280)
        h = cam.get("image_height", 720)

        return cls(
            K_left  = np.array(K_left_list,  dtype=np.float64).reshape(3, 3),
            D_left  = np.array(cam["D_left"], dtype=np.float64),
            K_right = np.array(K_right_list,  dtype=np.float64).reshape(3, 3),
            D_right = np.array(cam["D_right"], dtype=np.float64),
            R       = np.array(cam["R"],       dtype=np.float64).reshape(3, 3),
            T       = np.array(cam["T"],       dtype=np.float64).reshape(3, 1),
            image_size=(w, h),
        )

    @classmethod
    def _make_mock(cls) -> "StereoRectifier":
        """构造 Mock 实例：不做任何校正，直接透传。"""
        instance = object.__new__(cls)
        instance.calibration_available = False
        instance.Q = np.eye(4, dtype=np.float64)  # 单位矩阵，深度计算无效
        instance.image_size = (1280, 720)
        return instance

    def rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        对原始双目图像对进行立体校正。

        Returns
        -------
        (left_rect, right_rect) : 校正后的图像对（BGR）
        """
        if not self.calibration_available:
            return left.copy(), right.copy()

        left_rect  = cv2.remap(left,  self._map1_left,  self._map2_left,  cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self._map1_right, self._map2_right, cv2.INTER_LINEAR)
        return left_rect, right_rect
