"""
双目内外参标定
==============
使用棋盘格标定板，对左右双目相机进行立体标定，
计算内参矩阵（K, D）、旋转矩阵 R 和平移向量 T，
并将结果写入 config.yaml。

用法：
    # 1. 准备：打印棋盘格（默认 9×6 内角点），固定双目相机
    # 2. 采集：在不同位置拍摄 15~25 组图片对，保存到 calib_images/ 目录
    #          文件名格式：calib_images/left_001.png, calib_images/right_001.png
    # 3. 运行：
    python calibration/stereo_calibrate.py
    python calibration/stereo_calibrate.py --dir my_calib_images --cols 9 --rows 6

快捷键（采集模式）：
    s   — 保存当前帧到 calib_images/
    q   — 退出采集，开始标定
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
import yaml

CONFIG_PATH  = os.path.join(ROOT_DIR, "config.yaml")
DEFAULT_DIR  = os.path.join(ROOT_DIR, "calib_images")


# ──────────────────────────────────────────────────────────────────────────────
# 标定核心函数
# ──────────────────────────────────────────────────────────────────────────────

def calibrate(
    calib_dir: str,
    board_cols: int = 9,
    board_rows: int = 6,
    square_mm: float = 25.0,
):
    """
    对 calib_dir 中的左右图像对进行双目标定，将结果写入 config.yaml。

    Parameters
    ----------
    calib_dir   : 包含 left_*.png 和 right_*.png 的目录
    board_cols  : 棋盘格内角点列数
    board_rows  : 棋盘格内角点行数
    square_mm   : 格子边长（毫米）
    """
    print(f"\n📐 双目标定开始")
    print(f"   图像目录   : {calib_dir}")
    print(f"   棋盘格规格 : {board_cols}×{board_rows} 内角点，格子 {square_mm}mm")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    board_size = (board_cols, board_rows)

    # 准备 3D 物体点（Z=0 平面）
    objp = np.zeros((board_cols * board_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_mm

    left_files  = sorted(glob.glob(os.path.join(calib_dir, "left_*.png")))
    right_files = sorted(glob.glob(os.path.join(calib_dir, "right_*.png")))

    if not left_files:
        print(f"❌ 未找到标定图像（{calib_dir}/left_*.png）")
        return

    print(f"\n找到图像对 {len(left_files)} 组，开始检测角点…")

    obj_points  = []  # 3D 点（所有图片共用）
    img_pts_l   = []  # 左图 2D 角点
    img_pts_r   = []  # 右图 2D 角点
    image_size  = None

    for lf, rf in zip(left_files, right_files):
        img_l = cv2.imread(lf, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.imread(rf, cv2.IMREAD_GRAYSCALE)
        if img_l is None or img_r is None:
            print(f"  ⚠️  跳过（读取失败）: {os.path.basename(lf)}")
            continue

        if image_size is None:
            image_size = (img_l.shape[1], img_l.shape[0])

        ret_l, corners_l = cv2.findChessboardCorners(img_l, board_size, None)
        ret_r, corners_r = cv2.findChessboardCorners(img_r, board_size, None)

        if ret_l and ret_r:
            corners_l = cv2.cornerSubPix(img_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r = cv2.cornerSubPix(img_r, corners_r, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_pts_l.append(corners_l)
            img_pts_r.append(corners_r)
            print(f"  ✅ {os.path.basename(lf)}")
        else:
            print(f"  ❌ 角点检测失败: {os.path.basename(lf)}")

    if len(obj_points) < 5:
        print(f"\n❌ 有效图像对不足（{len(obj_points)} < 5），标定失败")
        return

    print(f"\n有效图像对: {len(obj_points)} 组，开始立体标定…")

    # 单目标定（初始化）
    _, K_l, D_l, _, _ = cv2.calibrateCamera(obj_points, img_pts_l, image_size, None, None)
    _, K_r, D_r, _, _ = cv2.calibrateCamera(obj_points, img_pts_r, image_size, None, None)

    # 立体标定
    rms, K_l, D_l, K_r, D_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_pts_l, img_pts_r,
        K_l, D_l, K_r, D_r,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    print(f"\n✅ 标定完成！RMS 重投影误差: {rms:.4f} px")
    if rms > 1.0:
        print("  ⚠️  RMS > 1.0，建议重新采集更多图像（25+ 组，不同角度和距离）")

    print(f"\n内参矩阵（左）:\n{K_l}")
    print(f"\n基线 T: {T.T} mm，|B| = {np.linalg.norm(T):.2f} mm")

    # ── 写入 config.yaml ──────────────────────────────────────────────────
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["camera"]["K_left"]  = K_l.flatten().tolist()
    cfg["camera"]["D_left"]  = D_l.flatten().tolist()
    cfg["camera"]["K_right"] = K_r.flatten().tolist()
    cfg["camera"]["D_right"] = D_r.flatten().tolist()
    cfg["camera"]["R"]       = R.flatten().tolist()
    cfg["camera"]["T"]       = T.flatten().tolist()
    cfg["camera"]["baseline_mm"] = float(np.linalg.norm(T))

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\n💾 标定结果已保存到 {CONFIG_PATH}")
    print("\n下一步：运行 calibration/extrinsic_calibrate.py 进行相机-世界外参标定")


# ──────────────────────────────────────────────────────────────────────────────
# 图像采集辅助（用相机实时采集棋盘格图像）
# ──────────────────────────────────────────────────────────────────────────────

def collect_images(calib_dir: str, cam_l: int, cam_r: int, board_cols: int, board_rows: int):
    """打开双目相机，实时显示角点检测结果，按 s 保存图像对。"""
    os.makedirs(calib_dir, exist_ok=True)
    board_size = (board_cols, board_rows)
    cap_l = cv2.VideoCapture(cam_l)
    cap_r = cv2.VideoCapture(cam_r)

    count = len(glob.glob(os.path.join(calib_dir, "left_*.png")))
    print(f"采集模式：已有 {count} 组，按 s 保存，q 开始标定")

    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            continue

        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)
        found_l, corners_l = cv2.findChessboardCorners(gray_l, board_size, None)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, board_size, None)

        vis_l = frame_l.copy()
        vis_r = frame_r.copy()
        cv2.drawChessboardCorners(vis_l, board_size, corners_l, found_l)
        cv2.drawChessboardCorners(vis_r, board_size, corners_r, found_r)

        status = "✅ 双目都检测到" if (found_l and found_r) else "❌ 未完整检测"
        color  = (0, 255, 0) if (found_l and found_r) else (0, 0, 255)
        cv2.putText(vis_l, f"{status}  count={count}  s=保存 q=完成",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        display = np.hstack([vis_l, vis_r])
        cv2.imshow("标定采集", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and found_l and found_r:
            count += 1
            cv2.imwrite(os.path.join(calib_dir, f"left_{count:03d}.png"),  frame_l)
            cv2.imwrite(os.path.join(calib_dir, f"right_{count:03d}.png"), frame_r)
            print(f"  💾 保存第 {count} 组")
        elif key in (ord('q'), 27):
            break

    cap_l.release()
    cap_r.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="双目立体标定工具")
    parser.add_argument("--dir",     default=DEFAULT_DIR, help="标定图像目录")
    parser.add_argument("--cols",    type=int, default=9,    help="棋盘格内角点列数")
    parser.add_argument("--rows",    type=int, default=6,    help="棋盘格内角点行数")
    parser.add_argument("--square",  type=float, default=25.0, help="格子边长（毫米）")
    parser.add_argument("--collect", action="store_true", help="先采集图像再标定")
    parser.add_argument("--camera",  nargs=2, type=int, default=[0, 1])
    args = parser.parse_args()

    if args.collect:
        collect_images(args.dir, args.camera[0], args.camera[1], args.cols, args.rows)

    calibrate(args.dir, args.cols, args.rows, args.square)


if __name__ == "__main__":
    main()
