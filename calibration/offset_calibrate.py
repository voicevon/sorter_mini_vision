"""
d_cam_offset 标定
=================
标定相机 Y 轴坐标与拨钩实际 Y 轴之间的固定偏置。

原理（来自 design_conclusion.md §五）：
    1. 在传送带上放置细标记点（牙签端点或标靶）
    2. 驱动滑台使拨钩尖端对齐该标记，记录步进位置 Y_hook（毫米）
    3. 相机拍照，点击图像中该标记点，反投影得到 Y_cam（毫米）
    4. d_cam_offset = Y_hook - Y_cam

写入 config.yaml 的 planner.d_cam_offset_mm。

用法：
    python calibration/offset_calibrate.py
    python calibration/offset_calibrate.py --image snap.png --y_hook 145.3
"""

import sys
import os
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import cv2
import numpy as np
import yaml

CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")


def pixel_to_world_y(
    px: float, py: float,
    K: np.ndarray,
    D: np.ndarray,
    T_cam2world: np.ndarray,
    work_distance_mm: float,
) -> float:
    """
    将左目图像中的像素坐标反投影到世界坐标系中的 Y 值（mm）。
    假设标记点在传送带平面（Z_world = 0）。
    """
    # 去畸变
    pts = np.array([[px, py]], dtype=np.float32).reshape(1, 1, 2)
    undist = cv2.undistortPoints(pts, K, D, P=K).reshape(2)
    u, v = undist

    # 在相机坐标系中，射线方向（Z_cam = 工作距离方向）
    cx = K[0, 2]; cy = K[1, 2]; fx = K[0, 0]; fy = K[1, 1]
    x_cam_norm = (u - cx) / fx  # 归一化相机坐标
    y_cam_norm = (v - cy) / fy

    # 用工作距离近似：标记点在 Z_cam = work_distance_mm
    Z_cam = work_distance_mm
    x_cam = x_cam_norm * Z_cam
    y_cam = y_cam_norm * Z_cam
    z_cam = Z_cam

    # 相机坐标 → 世界坐标
    p_cam = np.array([x_cam, y_cam, z_cam, 1.0])
    p_world = T_cam2world @ p_cam

    return float(p_world[1])  # Y 分量


def main():
    parser = argparse.ArgumentParser(description="d_cam_offset 标定工具")
    parser.add_argument("--image",  default=None,  help="标定图像路径（不指定则实时拍摄）")
    parser.add_argument("--y_hook", type=float, default=None,
                        help="拨钩对齐标记时的 Y 轴步进位置（毫米）")
    parser.add_argument("--camera", type=int, default=0, help="左目相机索引")
    args = parser.parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    K_list  = cfg["camera"].get("K_left", [])
    T_list  = cfg["camera"].get("T_cam2world", [])
    Z_work  = cfg["camera"].get("work_distance_mm", 800.0)

    if not K_list:
        print("❌ 请先完成双目内参标定")
        sys.exit(1)
    if not T_list:
        print("❌ 请先完成外参标定（extrinsic_calibrate.py）")
        sys.exit(1)

    K = np.array(K_list).reshape(3, 3)
    D = np.array(cfg["camera"].get("D_left", [0, 0, 0, 0, 0]))
    T_cam2world = np.array(T_list).reshape(4, 4)

    # ── 获取拨钩步进位置 ──────────────────────────────────────────────────
    if args.y_hook is None:
        print("\n📌 步骤 1：驱动滑台使拨钩对齐传送带上的标记点")
        print("         然后输入当前步进位置对应的 Y 轴毫米数：")
        while True:
            try:
                y_hook = float(input("  Y_hook [mm]: ").strip())
                break
            except ValueError:
                print("  请输入数字")
    else:
        y_hook = args.y_hook
        print(f"Y_hook = {y_hook} mm")

    # ── 获取图像 ──────────────────────────────────────────────────────────
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"❌ 无法读取图像: {args.image}")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(args.camera)
        print("\n📌 步骤 2：按空格拍摄当前标记点位置的图像")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("拍摄标定图像（空格=拍摄）", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                img = frame.copy()
                break
        cap.release()
        cv2.destroyAllWindows()

    # ── 点击标记点 ────────────────────────────────────────────────────────
    print("\n📌 步骤 3：点击图像中的标记点（牙签端点 / 标靶中心）")
    clicked: list[tuple] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear()
            clicked.append((x, y))
            vis = img.copy()
            cv2.circle(vis, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(vis, f"({x}, {y})", (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("点击标记点（Enter确认）", vis)

    cv2.imshow("点击标记点（Enter确认）", img)
    cv2.setMouseCallback("点击标记点（Enter确认）", on_mouse)
    print("  左键点击标记点，Enter 确认")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not clicked:
        print("❌ 未点击任何标记点")
        sys.exit(1)

    px, py = clicked[0]

    # ── 计算偏置 ──────────────────────────────────────────────────────────
    y_cam = pixel_to_world_y(px, py, K, D, T_cam2world, Z_work)
    d_cam_offset = y_hook - y_cam

    print(f"\n✅ 标定结果：")
    print(f"   像素坐标  : ({px}, {py})")
    print(f"   相机 Y    : {y_cam:.2f} mm")
    print(f"   拨钩 Y    : {y_hook:.2f} mm")
    print(f"   d_cam_offset = {d_cam_offset:.2f} mm")

    # ── 写入 config.yaml ──────────────────────────────────────────────────
    if "planner" not in cfg:
        cfg["planner"] = {}
    cfg["planner"]["d_cam_offset_mm"] = round(d_cam_offset, 2)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"\n💾 d_cam_offset = {d_cam_offset:.2f}mm 已保存到 {CONFIG_PATH}")
    print("\n✅ 所有标定完成！可以运行 python main.py 启动系统。")


if __name__ == "__main__":
    main()
