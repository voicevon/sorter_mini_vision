"""
双目图像采集线程
================
从双目相机（USB 或 MIPI）采集左右图像对，放入队列供识别线程消费。

Mock 模式（mock=True）：不需要真实相机，生成带绿色"芦笋"的合成图像对。
用于硬件不可用时的开发和测试。
"""

from __future__ import annotations
import logging
import queue
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FramePair:
    """一对原始双目图像。"""
    left:  np.ndarray    # BGR
    right: np.ndarray    # BGR
    timestamp_ms: int    # 采集时刻（系统时间）


class CaptureThread(threading.Thread):
    """
    双目图像采集线程。

    Parameters
    ----------
    out_queue : queue.Queue[FramePair]
        输出队列，满时丢弃最旧的帧（保持实时性）。
    camera_index_left : int
        左目相机的 VideoCapture 索引（默认 0）。
    camera_index_right : int
        右目相机的 VideoCapture 索引（默认 1）。
        某些双目相机以单设备输出左右拼接图像时，设 right=-1 并由本模块分割。
    mock : bool
        True = 不使用真实相机，生成合成图像对。
    fps : float
        目标帧率（Mock 模式下用 sleep 控制，真实模式由相机决定）。
    """

    def __init__(
        self,
        out_queue: "queue.Queue[FramePair]",
        camera_index_left: int = 0,
        camera_index_right: int = 1,
        mock: bool = True,
        fps: float = 5.0,
        image_size: tuple[int, int] = (1280, 720),
    ):
        super().__init__(name="CaptureThread", daemon=True)
        self._queue = out_queue
        self._idx_left  = camera_index_left
        self._idx_right = camera_index_right
        self._mock = mock
        self._fps  = fps
        self._image_size = image_size
        self._stop_event = threading.Event()
        self._frame_count = 0

    # ──────────────────────────────────────────────────────────────────────
    # 线程主循环
    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        if self._mock:
            logger.info("[采集] Mock 模式启动，帧率 %.1f fps", self._fps)
            self._run_mock()
        else:
            logger.info("[采集] 真实相机模式，设备 L=%d R=%d", self._idx_left, self._idx_right)
            self._run_real()

    def stop(self):
        self._stop_event.set()

    # ──────────────────────────────────────────────────────────────────────
    # Mock 模式：生成合成双目图像对
    # ──────────────────────────────────────────────────────────────────────

    def _run_mock(self):
        interval = 1.0 / self._fps
        rng = np.random.default_rng(seed=42)

        while not self._stop_event.is_set():
            t_start = time.monotonic()

            left, right = self._generate_mock_pair(rng)
            self._push(FramePair(left=left, right=right,
                                  timestamp_ms=int(time.time() * 1000)))

            elapsed = time.monotonic() - t_start
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)

    def _generate_mock_pair(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        生成一对合成双目图像：
        - 若干绿色细长矩形（模拟芦笋茎杆），大致平行于 X 轴，±15° 偏角
        - 部分茎杆略有叠压（用 z 偏移模拟不同亮度）
        - 右图相对左图水平平移（模拟视差）

        视差量与工作距离成反比，这里固定用 baseline=120mm, Z=800mm, f=1066px
        → 平均视差 ≈ 40px
        """
        w, h = self._image_size
        left = np.full((h, w, 3), (180, 180, 160), dtype=np.uint8)  # 浅灰背景

        # 随机生成 3~7 根芦笋
        n = rng.integers(3, 8)
        stems = []
        for _ in range(n):
            cx  = int(rng.uniform(w * 0.1, w * 0.9))
            cy  = int(rng.uniform(h * 0.2, h * 0.8))
            length = int(rng.uniform(h * 0.4, h * 0.7))
            diam_px = int(rng.uniform(6, 20))
            angle = rng.uniform(-15, 15)
            layer = int(rng.random() < 0.25)  # 25% 概率为叠层
            # 叠层芦笋颜色更浅（高位离光源近）
            g_val = int(rng.uniform(80, 120)) + layer * 40
            color = (int(g_val * 0.3), g_val, int(g_val * 0.4))
            stems.append((cx, cy, length, diam_px, angle, color, layer))

        # 先画底层
        for s in stems:
            cx, cy, length, diam, angle, color, layer = s
            if layer == 0:
                box = cv2.boxPoints(((cx, cy), (diam, length), angle)).astype(np.int32)
                cv2.fillPoly(left, [box], color)
        # 再画叠层（覆盖在上面）
        for s in stems:
            cx, cy, length, diam, angle, color, layer = s
            if layer == 1:
                cy_shifted = cy - 4  # 叠层在图像中位置略高
                box = cv2.boxPoints(((cx, cy_shifted), (diam, length), angle)).astype(np.int32)
                cv2.fillPoly(left, [box], color)

        # 右图 = 左图水平平移（模拟视差 35~45px）
        disparity_px = int(rng.uniform(35, 45))
        M = np.float32([[1, 0, -disparity_px], [0, 1, 0]])
        right = cv2.warpAffine(left, M, (w, h), borderValue=(180, 180, 160))

        # 加少量高斯噪声
        noise = rng.integers(-8, 8, left.shape, dtype=np.int16)
        left  = np.clip(left.astype(np.int16)  + noise, 0, 255).astype(np.uint8)
        right = np.clip(right.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 标注帧号（方便调试）
        cv2.putText(left,  f"MOCK L #{self._frame_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        cv2.putText(right, f"MOCK R #{self._frame_count}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        self._frame_count += 1

        return left, right

    # ──────────────────────────────────────────────────────────────────────
    # 真实相机模式
    # ──────────────────────────────────────────────────────────────────────

    def _run_real(self):
        cap_l = cv2.VideoCapture(self._idx_left)
        cap_r = cv2.VideoCapture(self._idx_right)

        if not cap_l.isOpened():
            logger.error("无法打开左目相机（index=%d）", self._idx_left)
            return
        if not cap_r.isOpened():
            logger.error("无法打开右目相机（index=%d）", self._idx_right)
            cap_l.release()
            return

        w, h = self._image_size
        for cap in (cap_l, cap_r):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        logger.info("[采集] 相机已打开，分辨率 %dx%d", w, h)

        try:
            while not self._stop_event.is_set():
                ok_l, frame_l = cap_l.read()
                ok_r, frame_r = cap_r.read()

                if not ok_l or not ok_r:
                    logger.warning("[采集] 读帧失败，重试…")
                    time.sleep(0.05)
                    continue

                self._push(FramePair(
                    left=frame_l,
                    right=frame_r,
                    timestamp_ms=int(time.time() * 1000),
                ))
        finally:
            cap_l.release()
            cap_r.release()
            logger.info("[采集] 相机已释放")

    # ──────────────────────────────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────────────────────────────

    def _push(self, frame_pair: FramePair):
        """将帧推入队列。队列满时丢弃最旧的帧，保持实时性。"""
        try:
            self._queue.put_nowait(frame_pair)
        except queue.Full:
            try:
                self._queue.get_nowait()  # 丢弃旧帧
            except queue.Empty:
                pass
            self._queue.put_nowait(frame_pair)
