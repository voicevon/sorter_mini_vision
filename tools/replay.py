"""
离线回放工具
============
回放已保存的图像对（左图 + 右图），重跑完整识别管线，
用于在无相机环境下调试和验证算法。

图像对保存格式（由 depth_viewer 的 's' 键生成）：
    debug_frames/<timestamp>_left.png
    debug_frames/<timestamp>_right.png（可选，无则使用左图）

用法：
    python tools/replay.py                            # 回放 debug_frames/ 目录
    python tools/replay.py --dir path/to/frames       # 指定目录
    python tools/replay.py --file left.png right.png  # 指定图片对
    python tools/replay.py --mock                     # 用 Mock 图像对回放

快捷键：
    空格 — 暂停 / 继续
    n   — 下一帧（暂停时有效）
    s   — 保存当前识别结果截图
    q   — 退出
"""

import sys
import os
import argparse
import glob
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np

from camera.capture import CaptureThread, FramePair
from camera.rectify import StereoRectifier
from vision.detector import AsparaguusDetector
from vision.sgbm import SGBMDepthEstimator
from vision.depth_integrator import DepthIntegrator
from vision.selector import build_snapshot
import queue

CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
SAVE_DIR    = os.path.join(ROOT_DIR, "debug_frames")


def draw_results(img: np.ndarray, snapshot) -> np.ndarray:
    """在图像上绘制识别结果叠加层。"""
    vis = img.copy()
    for asp in snapshot.asparagus_list:
        # 将世界坐标近似换回像素坐标（仅用于可视化，精度够用）
        # 此处使用 detector 内的估算
        color = (0, 100, 255) if asp.layer == 1 else (0, 220, 80)
        # 标注文字
        h, w = vis.shape[:2]
        # 用 y_center 和 x_center 估算像素位置（简化）
        px_to_mm = 800.0 / 1066.0
        u = int(w / 2 + asp.x_center / px_to_mm)
        v = int(h / 2 + asp.y_center / px_to_mm)
        u = max(10, min(u, w - 10))
        v = max(10, min(v, h - 10))

        cv2.circle(vis, (u, v), 8, color, -1)
        label = f"L{asp.layer} z={asp.z_top:.1f}mm d={asp.diameter_mm:.0f}mm"
        cv2.putText(vis, label, (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 目标芦笋额外标注
    if snapshot.rightmost_target:
        t = snapshot.rightmost_target
        px_to_mm = 800.0 / 1066.0
        u = int(vis.shape[1] / 2 + t.x_center / px_to_mm)
        v = int(vis.shape[0] / 2 + t.y_center / px_to_mm)
        u = max(10, min(u, vis.shape[1] - 10))
        v = max(10, min(v, vis.shape[0] - 10))
        cv2.circle(vis, (u, v), 14, (0, 0, 255), 3)
        cv2.putText(vis, "TARGET", (u + 16, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 信息栏
    info = (f"Count={snapshot.count}  Bottom={len(snapshot.bottom_layer_targets)}"
            f"  Target={'Y' if snapshot.rightmost_target else 'N'}")
    cv2.putText(vis, info, (10, vis.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return vis


def run_replay(frames: list[tuple], mock: bool):
    """主回放循环。frames 为 (left_img, right_img_or_None) 列表。"""
    rectifier  = StereoRectifier.from_config(CONFIG_PATH)
    sgbm       = SGBMDepthEstimator.from_config(CONFIG_PATH)
    detector   = AsparaguusDetector.from_config(CONFIG_PATH)
    integrator = DepthIntegrator.from_config(CONFIG_PATH)

    os.makedirs(SAVE_DIR, exist_ok=True)
    paused = False
    idx    = 0
    save_count = 0

    print(f"回放 {len(frames)} 帧。空格=暂停  n=下一帧  s=保存  q=退出")

    while idx < len(frames):
        left_raw, right_raw = frames[idx]
        if right_raw is None:
            right_raw = left_raw.copy()

        # ── 处理管线 ──────────────────────────────────────────────────────
        left_rect, right_rect = rectifier.rectify(left_raw, right_raw)

        if mock or not rectifier.calibration_available:
            disparity = SGBMDepthEstimator.make_mock_disparity(left_rect)
        else:
            disparity = sgbm.compute_disparity(left_rect, right_rect)

        poses = detector.detect(left_rect)
        if poses:
            integrator.integrate_mock(poses) if mock else integrator.integrate(
                poses, disparity, [detector.get_mask(left_rect)] * len(poses)
            )

        snapshot = build_snapshot(
            timestamp_ms         = int(time.time() * 1000),
            conveyor_position_mm = 0.0,
            asparagus_list       = poses,
        )

        # ── 可视化 ────────────────────────────────────────────────────────
        vis = draw_results(left_rect, snapshot)

        # 视差伪彩色
        d_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        d_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

        # 拼接
        display = np.hstack([vis, d_color])
        frame_label = f"帧 {idx+1}/{len(frames)}"
        if paused:
            frame_label += "  [暂停]"
        cv2.putText(display, frame_label, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow("离线回放 | replay", display)

        key = cv2.waitKey(1 if not paused else 50) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('n') and paused:
            idx = min(idx + 1, len(frames) - 1)
            continue
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(os.path.join(SAVE_DIR, f"replay_{ts}_vis.png"), display)
            save_count += 1
            print(f"💾 保存截图 #{save_count}")

        if not paused:
            idx += 1
            time.sleep(0.3)

    cv2.destroyAllWindows()
    print("回放结束。")


def load_frames_from_dir(directory: str) -> list[tuple]:
    """从目录加载图像对。"""
    left_files = sorted(glob.glob(os.path.join(directory, "*_left.png")))
    if not left_files:
        # 尝试加载所有 png
        all_pngs = sorted(glob.glob(os.path.join(directory, "*.png")))
        if len(all_pngs) >= 2:
            # 两两配对
            return [(cv2.imread(all_pngs[i]), cv2.imread(all_pngs[i+1]))
                    for i in range(0, len(all_pngs) - 1, 2)]
        return []

    frames = []
    for lf in left_files:
        rf = lf.replace("_left.png", "_right.png")
        left  = cv2.imread(lf)
        right = cv2.imread(rf) if os.path.exists(rf) else None
        if left is not None:
            frames.append((left, right))
    return frames


def generate_mock_frames(n: int = 10) -> list[tuple]:
    """生成 n 帧 Mock 图像对（用于完全离线测试）。"""
    q: queue.Queue[FramePair] = queue.Queue()
    capture = CaptureThread(out_queue=q, mock=True, fps=100.0)
    frames = []
    import threading
    t = threading.Thread(target=capture.run, daemon=True)
    t.start()
    for _ in range(n):
        fp = q.get(timeout=2.0)
        frames.append((fp.left, fp.right))
    capture.stop()
    return frames


def main():
    parser = argparse.ArgumentParser(description="识别管线离线回放工具")
    parser.add_argument("--dir",  default=SAVE_DIR, help="图像对目录")
    parser.add_argument("--file", nargs=2, metavar=("LEFT", "RIGHT"), help="单个图片对")
    parser.add_argument("--mock", action="store_true", help="使用 Mock 生成图像对")
    args = parser.parse_args()

    if args.mock:
        print("生成 Mock 图像对…")
        frames = generate_mock_frames(n=20)
    elif args.file:
        left  = cv2.imread(args.file[0])
        right = cv2.imread(args.file[1])
        if left is None:
            print(f"❌ 无法读取: {args.file[0]}")
            sys.exit(1)
        frames = [(left, right)]
    else:
        frames = load_frames_from_dir(args.dir)
        if not frames:
            print(f"⚠️  目录 {args.dir} 中未找到图像对，切换到 Mock 模式")
            frames = generate_mock_frames(n=10)

    run_replay(frames, mock=args.mock)


if __name__ == "__main__":
    main()
