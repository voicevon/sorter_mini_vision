"""
深度图可视化工具
================
实时显示 SGBM 视差图和伪彩色深度图，辅助调整 SGBM 参数。

用法：
    python tools/depth_viewer.py                         # Mock 模式
    python tools/depth_viewer.py --camera 0 1            # 真实双目相机
    python tools/depth_viewer.py --image left.png right.png  # 静态图片对

快捷键：
    s   — 保存当前帧（左图 + 右图 + 视差图）到 debug_frames/ 目录
    q/ESC — 退出
"""

import sys
import os
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np

from camera.capture import CaptureThread, FramePair
from camera.rectify import StereoRectifier
from vision.sgbm import SGBMDepthEstimator
import queue
import time


CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")
SAVE_DIR    = os.path.join(ROOT_DIR, "debug_frames")


def disparity_to_colormap(disparity: np.ndarray) -> np.ndarray:
    """视差图转伪彩色图（TURBO 色表，蓝=远 / 红=近）。"""
    d = disparity.copy()
    d[d <= 0] = 0
    d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)


def run_static(left_path: str, right_path: str):
    """用静态图片对运行一次深度估计并显示。"""
    left  = cv2.imread(left_path)
    right = cv2.imread(right_path)
    if left is None or right is None:
        print(f"❌ 无法读取图片: {left_path} / {right_path}")
        sys.exit(1)

    rectifier = StereoRectifier.from_config(CONFIG_PATH)
    sgbm      = SGBMDepthEstimator.from_config(CONFIG_PATH)

    left_rect, right_rect = rectifier.rectify(left, right)
    disparity = sgbm.compute_disparity(left_rect, right_rect)

    color_disp = disparity_to_colormap(disparity)

    # 拼接展示
    display = np.hstack([left_rect, color_disp])
    cv2.imshow("左图 | 视差图(伪彩色)", display)
    print("按任意键关闭…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_live(mock: bool, camera_left: int, camera_right: int):
    """实时深度图显示（相机或 Mock 模式）。"""
    frame_queue: queue.Queue[FramePair] = queue.Queue(maxsize=2)
    rectifier = StereoRectifier.from_config(CONFIG_PATH)
    sgbm      = SGBMDepthEstimator.from_config(CONFIG_PATH)

    capture = CaptureThread(
        out_queue          = frame_queue,
        camera_index_left  = camera_left,
        camera_index_right = camera_right,
        mock               = mock,
        fps                = 5.0,
    )
    capture.start()

    os.makedirs(SAVE_DIR, exist_ok=True)
    save_count = 0
    print("实时深度查看器启动。快捷键：s=保存帧  q/ESC=退出")

    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        left_rect, right_rect = rectifier.rectify(frame.left, frame.right)

        if mock or not rectifier.calibration_available:
            disparity = SGBMDepthEstimator.make_mock_disparity(left_rect)
        else:
            disparity = sgbm.compute_disparity(left_rect, right_rect)

        color_disp = disparity_to_colormap(disparity)

        # 添加极线（辅助检查校正质量）
        for y_line in range(0, left_rect.shape[0], 40):
            cv2.line(left_rect,  (0, y_line), (left_rect.shape[1], y_line),  (0, 255, 0), 1)
            cv2.line(color_disp, (0, y_line), (color_disp.shape[1], y_line), (0, 255, 0), 1)

        label_l = "左图 (立体校正后)"
        label_d = "视差图 (TURBO: 蓝=远 红=近)"
        if mock:
            label_l += " [MOCK]"
        cv2.putText(left_rect,  label_l, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)
        cv2.putText(color_disp, label_d, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 1)

        display = np.hstack([left_rect, color_disp])
        cv2.imshow("深度查看器 | depth_viewer", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ts = int(time.time())
            cv2.imwrite(os.path.join(SAVE_DIR, f"{ts}_left.png"),     left_rect)
            cv2.imwrite(os.path.join(SAVE_DIR, f"{ts}_disp.png"),     color_disp)
            np.save(os.path.join(SAVE_DIR, f"{ts}_disparity.npy"), disparity)
            save_count += 1
            print(f"💾 已保存帧 #{save_count} 到 {SAVE_DIR}/")

    capture.stop()
    cv2.destroyAllWindows()
    print("深度查看器已退出。")


def main():
    parser = argparse.ArgumentParser(description="SGBM 深度图可视化工具")
    parser.add_argument("--mock",   action="store_true", help="Mock 模式（无需相机）")
    parser.add_argument("--camera", nargs=2, type=int, default=[0, 1],
                        metavar=("LEFT_IDX", "RIGHT_IDX"), help="相机设备索引")
    parser.add_argument("--image",  nargs=2, metavar=("LEFT_IMG", "RIGHT_IMG"),
                        help="静态图片对（替代实时模式）")
    args = parser.parse_args()

    if args.image:
        run_static(args.image[0], args.image[1])
    else:
        run_live(args.mock, args.camera[0], args.camera[1])


if __name__ == "__main__":
    main()
