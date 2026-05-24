"""
相机-世界外参标定
==================
在传送带平面上放置已知坐标的标定板，
通过 solvePnP 计算相机到世界坐标系的变换矩阵 T_cam2world，
写入 config.yaml。

前置条件：
    1. 双目内参标定已完成（config.yaml 中 K_left 非空）
    2. 传送带上放置标定板，已测量各标定点的世界坐标（X, Y, Z=0）

标定板布置建议：
    - 在传送带平面上放 4~6 个已知位置的标记点（如棋盘格角点或专用标靶）
    - 用卷尺精确测量每个点相对于世界坐标系原点的位置（单位 mm）
    - 世界坐标系原点：双目相机光心正下方在传送带平面的投影点
    - X 轴：沿传送带方向，Y 轴：垂直传送带向右，Z=0：传送带平面

用法：
    python calibration/extrinsic_calibrate.py
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


def calibrate_extrinsic(
    world_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    使用 solvePnP 计算相机外参（旋转 + 平移）。

    Parameters
    ----------
    world_points_mm : (N, 3) float32，各标定点在世界坐标系中的位置（毫米）
    image_points_px : (N, 2) float32，对应的图像像素坐标
    K               : 3×3 相机内参矩阵
    D               : 畸变系数

    Returns
    -------
    T_cam2world : 4×4 齐次变换矩阵（相机坐标 → 世界坐标）
    """
    ret, rvec, tvec = cv2.solvePnP(
        world_points_mm.astype(np.float32),
        image_points_px.astype(np.float32),
        K, D,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ret:
        raise RuntimeError("solvePnP 失败，请检查对应点是否正确")

    R, _ = cv2.Rodrigues(rvec)  # 旋转向量 → 旋转矩阵

    # 构造相机到世界的 4×4 变换矩阵
    # 注意：solvePnP 输出的是世界→相机变换，需要取逆
    R_cw = R           # R 相机坐标系相对世界
    t_cw = tvec.flatten()
    # 逆变换：T_world2cam → T_cam2world
    R_wc = R_cw.T
    t_wc = -R_wc @ t_cw

    T = np.eye(4)
    T[:3, :3] = R_wc
    T[:3,  3] = t_wc

    return T


def interactive_collect(img_bgr: np.ndarray) -> list[tuple[float, float]]:
    """
    在图像上交互式点击标定点（鼠标左键），返回像素坐标列表。
    右键删除最后一个点，按 Enter 确认。
    """
    points = []
    vis = img_bgr.copy()

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(len(points)), (x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("点击标定点", vis)
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            # 重绘
            vis[:] = img_bgr[:]
            for i, (px, py) in enumerate(points):
                cv2.circle(vis, (int(px), int(py)), 6, (0, 255, 0), -1)
                cv2.putText(vis, str(i + 1), (int(px) + 8, int(py)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("点击标定点", vis)

    cv2.imshow("点击标定点", vis)
    cv2.setMouseCallback("点击标定点", on_mouse)

    print("  鼠标左键点击标定点（按标定板上的编号顺序）")
    print("  鼠标右键删除最后一个点，Enter 确认")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, ord('\r'), ord('q')):  # Enter 或 q
            break

    cv2.destroyAllWindows()
    return points


def main():
    parser = argparse.ArgumentParser(description="相机-世界外参标定")
    parser.add_argument("--image", default=None, help="标定用左目图像（不指定则从相机拍摄）")
    parser.add_argument("--camera", type=int, default=0, help="左目相机索引")
    args = parser.parse_args()

    # ── 加载内参 ──────────────────────────────────────────────────────────
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    K_list = cfg["camera"].get("K_left", [])
    if not K_list:
        print("❌ 请先完成双目内参标定（stereo_calibrate.py）")
        sys.exit(1)

    K = np.array(K_list).reshape(3, 3)
    D = np.array(cfg["camera"].get("D_left", [0, 0, 0, 0, 0]))

    # ── 获取图像 ──────────────────────────────────────────────────────────
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"❌ 无法读取图像: {args.image}")
            sys.exit(1)
    else:
        cap = cv2.VideoCapture(args.camera)
        print("按空格拍摄标定图像…")
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

    # ── 输入世界坐标（手动） ──────────────────────────────────────────────
    print("\n请输入各标定点的世界坐标（毫米）。")
    print("按照你在图像上点击的顺序依次输入 X Y（Z=0，传送带平面）。")
    print("输入示例：  0 0  （原点）   或   200 100  （X=200mm, Y=100mm）\n")

    # ── 交互点击图像点 ────────────────────────────────────────────────────
    print("📌 步骤 1：在图像上依次点击标定点（与世界坐标顺序一致）")
    image_pts_raw = interactive_collect(img)
    n = len(image_pts_raw)
    if n < 4:
        print(f"❌ 需要至少 4 个标定点，当前只有 {n} 个")
        sys.exit(1)

    print(f"\n📌 步骤 2：输入 {n} 个标定点的世界坐标（X Y，以空格分隔）")
    world_pts = []
    for i in range(n):
        while True:
            try:
                raw = input(f"  点 {i+1} (px={image_pts_raw[i][0]:.0f},{image_pts_raw[i][1]:.0f}) → 世界 X Y [mm]: ")
                x, y = map(float, raw.strip().split())
                world_pts.append([x, y, 0.0])
                break
            except ValueError:
                print("  格式错误，请输入两个数字，例如：200 100")

    world_pts_np = np.array(world_pts, dtype=np.float32)
    image_pts_np = np.array(image_pts_raw, dtype=np.float32)

    # ── 计算外参 ──────────────────────────────────────────────────────────
    T_cam2world = calibrate_extrinsic(world_pts_np, image_pts_np, K, D)
    print(f"\n✅ T_cam2world:\n{T_cam2world}")

    # 验证重投影误差
    errors = []
    for wp, ip in zip(world_pts_np, image_pts_np):
        p_cam = T_cam2world[:3, :3].T @ (wp - T_cam2world[:3, 3])  # 简化逆变换验证
        proj, _ = cv2.projectPoints(
            wp.reshape(1, 3), cv2.Rodrigues(T_cam2world[:3, :3].T)[0],
            -T_cam2world[:3, :3].T @ T_cam2world[:3, 3],
            K, D,
        )
        err = np.linalg.norm(proj.flatten() - ip)
        errors.append(err)
    print(f"重投影误差：平均 {np.mean(errors):.2f}px，最大 {np.max(errors):.2f}px")

    # ── 写入 config.yaml ──────────────────────────────────────────────────
    cfg["camera"]["T_cam2world"] = T_cam2world.flatten().tolist()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"\n💾 T_cam2world 已保存到 {CONFIG_PATH}")
    print("\n下一步：运行 calibration/offset_calibrate.py 进行 d_cam_offset 标定")


if __name__ == "__main__":
    main()
