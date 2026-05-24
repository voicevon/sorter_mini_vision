#!/usr/bin/env python3
"""
芦笋分拣视觉系统 — 主程序
==========================
启动 4 个守护线程，通过 3 个 Queue 解耦：

    [采集] → queue_frames → [识别] → queue_snapshots → [规划] → queue_commands → [BLE]

用法：
    # Mock 模式（无硬件）：
    python main.py --mock

    # 真实模式（需要相机和 BLE 设备）：
    python main.py

    # 查看所有选项：
    python main.py --help
"""

import argparse
import logging
import queue
import signal
import sys
import time

import yaml

from camera.capture import CaptureThread, FramePair
from camera.rectify import StereoRectifier
from comm.ble_client import BLEClientThread
from models import HookCommand, SceneSnapshot
from planner.hook_planner import HookPlanner
from vision.depth_integrator import DepthIntegrator
from vision.detector import AsparaguusDetector
from vision.selector import build_snapshot
from vision.sgbm import SGBMDepthEstimator

CONFIG_PATH = "config.yaml"

# ──────────────────────────────────────────────────────────────────────────────
# 日志配置
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO"):
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format=fmt,
                        datefmt="%H:%M:%S")

logger = logging.getLogger("main")


# ──────────────────────────────────────────────────────────────────────────────
# 识别线程（在主进程中作为函数运行，由 threading.Thread 包装）
# ──────────────────────────────────────────────────────────────────────────────

def recognition_worker(
    in_queue:  "queue.Queue[FramePair]",
    out_queue: "queue.Queue[SceneSnapshot]",
    rectifier: StereoRectifier,
    sgbm:      SGBMDepthEstimator,
    detector:  AsparaguusDetector,
    integrator: DepthIntegrator,
    mock: bool,
    stop_flag: list[bool],
):
    """
    识别线程主函数：
    1. 从 in_queue 取图像对
    2. 立体校正
    3. SGBM 视差计算（或 Mock 伪深度）
    4. 芦笋检测（颜色分割 + 轮廓）
    5. 深度集成
    6. 目标选择，构建 SceneSnapshot
    7. 推入 out_queue
    """
    logger.info("[识别] 线程启动 mock=%s", mock)
    frame_count = 0

    while not stop_flag[0]:
        try:
            frame: FramePair = in_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        t0 = time.monotonic()

        # ── 立体校正 ──────────────────────────────────────────────────────
        left_rect, right_rect = rectifier.rectify(frame.left, frame.right)

        # ── 深度估计 ──────────────────────────────────────────────────────
        if mock or not rectifier.calibration_available:
            disparity = SGBMDepthEstimator.make_mock_disparity(left_rect)
        else:
            disparity = sgbm.compute_disparity(left_rect, right_rect)

        # ── 芦笋检测 ──────────────────────────────────────────────────────
        poses = detector.detect(left_rect)

        # 提取各连通域掩码（detector 内部已计算，需要重新获取）
        # 简化处理：用全图掩码统一采样（精度略低，后续可优化）
        mask = detector.get_mask(left_rect)

        # ── 深度集成 ──────────────────────────────────────────────────────
        if poses:
            if mock:
                integrator.integrate_mock(poses)
            else:
                # 为每根芦笋重建独立掩码（简化版：用全局绿色掩码）
                contour_masks = [mask] * len(poses)
                integrator.integrate(poses, disparity, contour_masks)

        # ── 目标选择 ──────────────────────────────────────────────────────
        snapshot = build_snapshot(
            timestamp_ms         = frame.timestamp_ms,
            conveyor_position_mm = 0.0,  # TODO: 接入编码器读数
            asparagus_list       = poses,
        )

        # ── 推入输出队列 ──────────────────────────────────────────────────
        try:
            out_queue.put_nowait(snapshot)
        except queue.Full:
            try:
                out_queue.get_nowait()
            except queue.Empty:
                pass
            out_queue.put_nowait(snapshot)

        elapsed_ms = (time.monotonic() - t0) * 1000
        frame_count += 1
        logger.debug("[识别] 帧 #%d 耗时 %.1fms，识别 %d 根",
                     frame_count, elapsed_ms, snapshot.count)
        in_queue.task_done()

    logger.info("[识别] 线程退出")


def planning_worker(
    in_queue:  "queue.Queue[SceneSnapshot]",
    out_queue: "queue.Queue[HookCommand]",
    planner:   HookPlanner,
    stop_flag: list[bool],
):
    """规划线程：SceneSnapshot → HookCommand。"""
    logger.info("[规划] 线程启动")
    while not stop_flag[0]:
        try:
            snapshot: SceneSnapshot = in_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        cmd = planner.plan(snapshot)
        if cmd is not None:
            try:
                out_queue.put_nowait(cmd)
            except queue.Full:
                logger.warning("[规划] 指令队列已满，丢弃旧指令")
                try:
                    out_queue.get_nowait()
                except queue.Empty:
                    pass
                out_queue.put_nowait(cmd)

        in_queue.task_done()

    logger.info("[规划] 线程退出")


# ──────────────────────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="芦笋分拣视觉系统")
    parser.add_argument("--mock",    action="store_true", help="Mock 模式（无需硬件）")
    parser.add_argument("--config",  default=CONFIG_PATH,  help="配置文件路径")
    parser.add_argument("--log",     default="INFO",        help="日志级别 DEBUG/INFO/WARNING")
    parser.add_argument("--fps",     type=float, default=2.0, help="Mock 采集帧率")
    args = parser.parse_args()

    setup_logging(args.log)
    logger.info("=" * 60)
    logger.info("芦笋分拣视觉系统启动  mock=%s  config=%s", args.mock, args.config)
    logger.info("=" * 60)

    # ── 加载配置 ──────────────────────────────────────────────────────────
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cam_cfg = cfg.get("camera", {})
    ble_cfg = cfg.get("ble",    {})

    # ── 构建各模块 ────────────────────────────────────────────────────────
    rectifier  = StereoRectifier.from_config(args.config)
    sgbm       = SGBMDepthEstimator.from_config(args.config)
    detector   = AsparaguusDetector.from_config(args.config)
    integrator = DepthIntegrator.from_config(args.config)
    planner    = HookPlanner.from_config(args.config)

    # ── 队列 ──────────────────────────────────────────────────────────────
    queue_frames    : "queue.Queue[FramePair]"     = queue.Queue(maxsize=2)
    queue_snapshots : "queue.Queue[SceneSnapshot]" = queue.Queue(maxsize=5)
    queue_commands  : "queue.Queue[HookCommand]"   = queue.Queue(maxsize=10)

    # 共享停止标志
    stop_flag: list[bool] = [False]

    # ── 启动线程 ──────────────────────────────────────────────────────────
    import threading

    t_capture = CaptureThread(
        out_queue          = queue_frames,
        camera_index_left  = cam_cfg.get("index_left",  0),
        camera_index_right = cam_cfg.get("index_right", 1),
        mock               = args.mock,
        fps                = args.fps,
        image_size         = (cam_cfg.get("image_width", 1280),
                               cam_cfg.get("image_height", 720)),
    )

    t_recognition = threading.Thread(
        target=recognition_worker,
        name="RecognitionThread",
        daemon=True,
        kwargs=dict(
            in_queue   = queue_frames,
            out_queue  = queue_snapshots,
            rectifier  = rectifier,
            sgbm       = sgbm,
            detector   = detector,
            integrator = integrator,
            mock       = args.mock,
            stop_flag  = stop_flag,
        ),
    )

    t_planning = threading.Thread(
        target=planning_worker,
        name="PlanningThread",
        daemon=True,
        kwargs=dict(
            in_queue  = queue_snapshots,
            out_queue = queue_commands,
            planner   = planner,
            stop_flag = stop_flag,
        ),
    )

    t_ble = BLEClientThread(
        in_queue        = queue_commands,
        device_address  = ble_cfg.get("device_address",   ""),
        service_uuid    = ble_cfg.get("service_uuid",      ""),
        char_write_uuid = ble_cfg.get("char_write_uuid",   ""),
        char_notify_uuid= ble_cfg.get("char_notify_uuid",  ""),
        ack_timeout_sec = ble_cfg.get("ack_timeout_sec",   5.0),
        mock            = args.mock,
    )

    # ── 信号处理（Ctrl+C 优雅退出）────────────────────────────────────────
    def _shutdown(signum, frame):
        logger.info("\n收到退出信号，正在停止…")
        stop_flag[0] = True
        t_capture.stop()
        t_ble.stop()

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── 启动 ──────────────────────────────────────────────────────────────
    t_capture.start()
    t_recognition.start()
    t_planning.start()
    t_ble.start()

    logger.info("所有线程已启动。按 Ctrl+C 退出。")

    # 主线程等待
    try:
        while not stop_flag[0]:
            time.sleep(1.0)
            # 定期打印 BLE 统计
            stats = t_ble.stats
            if stats["sent"] > 0:
                logger.info(
                    "[统计] BLE sent=%d ack_ok=%d timeout=%d error=%d",
                    stats["sent"], stats["ack_ok"],
                    stats["ack_timeout"], stats["error"],
                )
    except KeyboardInterrupt:
        _shutdown(None, None)

    # 等待线程退出
    for t in (t_recognition, t_planning):
        t.join(timeout=3.0)

    logger.info("系统已停止。")
    sys.exit(0)


if __name__ == "__main__":
    main()
