#!/usr/bin/env python3
"""
HSV 阈值调参工具
================
用途：在真实芦笋图像上，通过拖动滑块实时调整绿色 HSV 阈值，
      找到最佳参数后按 's' 保存到 config.yaml。

用法：
    python tools/hsv_tuner.py                         # 使用内置测试图案
    python tools/hsv_tuner.py path/to/asparagus.jpg   # 使用指定图片

快捷键：
    s     — 保存当前阈值到 config.yaml
    r     — 重置为 config.yaml 中的当前值
    q/ESC — 退出
"""

import sys
import os
import cv2
import numpy as np
import yaml

# 找到项目根目录（本文件在 tools/ 下，根目录是上一级）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT_DIR, "config.yaml")

WINDOW_SRC = "原始图像 (HSV Tuner)"
WINDOW_MASK = "掩码结果 (绿色提取)"
WINDOW_OVERLAY = "叠加预览 (绿色高亮)"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_hsv_to_config(h_min, s_min, v_min, h_max, s_max, v_max):
    config = load_config()
    config["detection"]["hsv_lower"] = [int(h_min), int(s_min), int(v_min)]
    config["detection"]["hsv_upper"] = [int(h_max), int(s_max), int(v_max)]
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"\n✅ 已保存到 {CONFIG_PATH}")
    print(f"   hsv_lower: [{h_min}, {s_min}, {v_min}]")
    print(f"   hsv_upper: [{h_max}, {s_max}, {v_max}]")


def make_test_image() -> np.ndarray:
    """生成一张测试图：绿色圆柱（模拟芦笋截面）在浅色背景上。"""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # 浅灰背景

    # 画几根"芦笋"（绿色细长矩形，略有旋转）
    asparagus_color_bgr = (34, 139, 34)   # 深绿
    light_green_bgr     = (50, 205, 50)   # 亮绿
    tip_bgr             = (80, 160, 80)   # 嫩尖部分

    stems = [
        # (center, (w, h), angle, color)
        ((200, 240), (18, 200), -5,  asparagus_color_bgr),
        ((240, 240), (16, 210),  2,  light_green_bgr),
        ((280, 235), (14, 195),  8,  asparagus_color_bgr),
        ((320, 245), (20, 205), -3,  tip_bgr),
        ((360, 238), (15, 200),  5,  light_green_bgr),
    ]
    for center, size, angle, color in stems:
        box = cv2.boxPoints((center, size, angle)).astype(np.int32)
        cv2.fillPoly(img, [box], color)
        # 加一点纹理噪声
        noise_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(noise_mask, [box], 255)
        noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise * (noise_mask[:, :, None] > 0), 0, 255).astype(np.uint8)

    # 在右下角加文字说明
    cv2.putText(img, "TEST IMAGE (no real asparagus photo)", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    return img


def nothing(_):
    """Trackbar 回调占位符。"""
    pass


def run_tuner(image_path: str | None):
    # ── 加载图片 ──────────────────────────────────────────────
    if image_path and os.path.exists(image_path):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"❌ 无法读取图片: {image_path}")
            sys.exit(1)
        print(f"📷 使用图片: {image_path}")
    else:
        if image_path:
            print(f"⚠️  找不到图片 '{image_path}'，使用内置测试图案")
        else:
            print("ℹ️  未指定图片，使用内置测试图案")
        img_bgr = make_test_image()

    # 限制显示尺寸（宽度最大 800px）
    h, w = img_bgr.shape[:2]
    if w > 800:
        scale = 800 / w
        img_bgr = cv2.resize(img_bgr, (800, int(h * scale)))

    # ── 读取 config.yaml 初始值 ───────────────────────────────
    try:
        cfg = load_config()
        det = cfg.get("detection", {})
        lo = det.get("hsv_lower", [35, 40, 40])
        hi = det.get("hsv_upper", [85, 255, 255])
        init = dict(h_min=lo[0], s_min=lo[1], v_min=lo[2],
                    h_max=hi[0], s_max=hi[1], v_max=hi[2])
    except Exception as e:
        print(f"⚠️  读取 config.yaml 失败（{e}），使用默认值")
        init = dict(h_min=35, s_min=40, v_min=40,
                    h_max=85, s_max=255, v_max=255)

    # ── 创建窗口与 Trackbar ───────────────────────────────────
    cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_MASK, img_bgr.shape[1] * 2, img_bgr.shape[0])

    bars = [
        ("H_min", init["h_min"], 179),
        ("S_min", init["s_min"], 255),
        ("V_min", init["v_min"], 255),
        ("H_max", init["h_max"], 179),
        ("S_max", init["s_max"], 255),
        ("V_max", init["v_max"], 255),
    ]
    for name, val, max_val in bars:
        cv2.createTrackbar(name, WINDOW_MASK, val, max_val, nothing)

    print("\n📌 快捷键：")
    print("   s   — 保存阈值到 config.yaml")
    print("   r   — 重置为当前 config.yaml 值")
    print("   q / ESC — 退出\n")

    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    while True:
        # 读取滑块值
        h_min = cv2.getTrackbarPos("H_min", WINDOW_MASK)
        s_min = cv2.getTrackbarPos("S_min", WINDOW_MASK)
        v_min = cv2.getTrackbarPos("V_min", WINDOW_MASK)
        h_max = cv2.getTrackbarPos("H_max", WINDOW_MASK)
        s_max = cv2.getTrackbarPos("S_max", WINDOW_MASK)
        v_max = cv2.getTrackbarPos("V_max", WINDOW_MASK)

        # 生成掩码
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img_hsv, lower, upper)

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        # 叠加预览：绿色高亮
        overlay = img_bgr.copy()
        green_tint = np.zeros_like(img_bgr)
        green_tint[:, :, 1] = 120  # 绿色通道
        overlay = np.where(mask_clean[:, :, None] > 0,
                           cv2.addWeighted(overlay, 0.5, green_tint, 0.5, 0),
                           overlay)

        # 在叠加图上显示当前参数
        param_text = (f"H:[{h_min},{h_max}]  "
                      f"S:[{s_min},{s_max}]  "
                      f"V:[{v_min},{v_max}]")
        cv2.putText(overlay, param_text, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "s=save  r=reset  q=quit", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # 拼接展示：左=叠加预览，右=掩码
        mask_3ch = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
        display = np.hstack([overlay, mask_3ch])
        cv2.imshow(WINDOW_MASK, display)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):  # q 或 ESC
            break
        elif key == ord('s'):
            save_hsv_to_config(h_min, s_min, v_min, h_max, s_max, v_max)
        elif key == ord('r'):
            # 重置为 config 当前值
            try:
                cfg = load_config()
                lo = cfg["detection"]["hsv_lower"]
                hi = cfg["detection"]["hsv_upper"]
                cv2.setTrackbarPos("H_min", WINDOW_MASK, lo[0])
                cv2.setTrackbarPos("S_min", WINDOW_MASK, lo[1])
                cv2.setTrackbarPos("V_min", WINDOW_MASK, lo[2])
                cv2.setTrackbarPos("H_max", WINDOW_MASK, hi[0])
                cv2.setTrackbarPos("S_max", WINDOW_MASK, hi[1])
                cv2.setTrackbarPos("V_max", WINDOW_MASK, hi[2])
                print("🔄 已重置为 config.yaml 当前值")
            except Exception as e:
                print(f"⚠️  重置失败: {e}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_tuner(img_path)
