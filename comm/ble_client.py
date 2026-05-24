"""
BLE 蓝牙通信线程
================
将 HookCommand 通过 BLE（Bluetooth Low Energy）发送给下位机，
并等待 ACK 确认。

使用 bleak 库（异步），在专用线程中运行独立的 asyncio 事件循环。

Mock 模式（mock=True）：不建立真实 BLE 连接，将指令打印到日志，
用于硬件不可用时的端到端测试。
"""

from __future__ import annotations
import asyncio
import json
import logging
import queue
import threading
import time

logger = logging.getLogger(__name__)

# 避免在没有安装 bleak 时导入失败
try:
    from bleak import BleakClient, BleakScanner
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False
    logger.warning("bleak 未安装，BLE 功能不可用。运行 pip install bleak 安装。")

from models import HookCommand


class BLEClientThread(threading.Thread):
    """
    BLE 通信线程。

    从 in_queue 取 HookCommand，通过 BLE Write Characteristic 发送 JSON，
    等待 Notify Characteristic 回传 ACK。

    Parameters
    ----------
    in_queue : queue.Queue[HookCommand]
        输入队列（来自规划线程）
    device_address : str
        下位机 BLE MAC 地址，如 "AA:BB:CC:DD:EE:FF"
    service_uuid : str
        BLE 服务 UUID
    char_write_uuid : str
        写特征 UUID（发送指令）
    char_notify_uuid : str
        通知特征 UUID（接收 ACK）
    ack_timeout_sec : float
        等待 ACK 超时（秒）
    mock : bool
        True = 打印模式，不建立真实连接
    """

    def __init__(
        self,
        in_queue: "queue.Queue[HookCommand]",
        device_address: str = "",
        service_uuid: str = "",
        char_write_uuid: str = "",
        char_notify_uuid: str = "",
        ack_timeout_sec: float = 5.0,
        mock: bool = True,
    ):
        super().__init__(name="BLEClientThread", daemon=True)
        self._queue          = in_queue
        self._device_address = device_address
        self._svc_uuid       = service_uuid
        self._write_uuid     = char_write_uuid
        self._notify_uuid    = char_notify_uuid
        self._ack_timeout    = ack_timeout_sec
        self._mock           = mock or not BLEAK_AVAILABLE or not device_address
        self._stop_event     = threading.Event()
        self._stats = {"sent": 0, "ack_ok": 0, "ack_timeout": 0, "error": 0}

    def stop(self):
        self._stop_event.set()

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    # ──────────────────────────────────────────────────────────────────────

    def run(self):
        if self._mock:
            logger.info("[BLE] Mock 模式，指令将打印到日志")
            self._run_mock()
        else:
            logger.info("[BLE] 真实模式，连接设备 %s", self._device_address)
            asyncio.run(self._run_real())

    # ──────────────────────────────────────────────────────────────────────
    # Mock 模式
    # ──────────────────────────────────────────────────────────────────────

    def _run_mock(self):
        while not self._stop_event.is_set():
            try:
                cmd: HookCommand = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            payload = json.dumps(cmd.to_dict())
            logger.info("[BLE MOCK] >> SEND: %s", payload)
            time.sleep(0.1)  # 模拟通信延迟
            # 模拟 ACK 回复
            ack = {"ack": cmd.seq, "status": "DONE"}
            logger.info("[BLE MOCK] << ACK : %s", json.dumps(ack))
            self._stats["sent"]  += 1
            self._stats["ack_ok"] += 1
            self._queue.task_done()

    # ──────────────────────────────────────────────────────────────────────
    # 真实 BLE 模式（asyncio）
    # ──────────────────────────────────────────────────────────────────────

    async def _run_real(self):
        """在独立 asyncio 事件循环中运行 BLE 通信。"""
        async with BleakClient(self._device_address) as client:
            logger.info("[BLE] 已连接 %s", self._device_address)

            # ACK 信号（asyncio.Event）：Notify 回调设置
            ack_event = asyncio.Event()
            ack_seq_received: list[int] = [-1]

            def on_notify(_: int, data: bytearray):
                try:
                    resp = json.loads(data.decode())
                    ack_seq_received[0] = resp.get("ack", -1)
                    status = resp.get("status", "?")
                    logger.info("[BLE] << ACK seq=%d status=%s",
                                ack_seq_received[0], status)
                    ack_event.set()
                except Exception as e:
                    logger.warning("[BLE] 解析 ACK 失败: %s", e)

            await client.start_notify(self._notify_uuid, on_notify)

            while not self._stop_event.is_set():
                # 从队列取指令（非阻塞，用 run_in_executor）
                loop = asyncio.get_event_loop()
                try:
                    cmd = await loop.run_in_executor(
                        None, lambda: self._queue.get(timeout=0.5)
                    )
                except queue.Empty:
                    continue

                payload = json.dumps(cmd.to_dict()).encode()
                try:
                    ack_event.clear()
                    await client.write_gatt_char(self._write_uuid, payload)
                    self._stats["sent"] += 1
                    logger.info("[BLE] >> SEND seq=%d y=%.1f z=%.1f",
                                cmd.seq, cmd.y_mm, cmd.z_mm)

                    # 等待 ACK
                    try:
                        await asyncio.wait_for(ack_event.wait(), timeout=self._ack_timeout)
                        if ack_seq_received[0] == cmd.seq:
                            self._stats["ack_ok"] += 1
                        else:
                            logger.warning("[BLE] ACK seq 不匹配 expected=%d got=%d",
                                           cmd.seq, ack_seq_received[0])
                    except asyncio.TimeoutError:
                        self._stats["ack_timeout"] += 1
                        logger.error("[BLE] ACK 超时 seq=%d", cmd.seq)

                except Exception as e:
                    self._stats["error"] += 1
                    logger.error("[BLE] 发送失败: %s", e)
                finally:
                    self._queue.task_done()

            await client.stop_notify(self._notify_uuid)
            logger.info("[BLE] 连接已断开")

    # ──────────────────────────────────────────────────────────────────────
    # 辅助工具
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    async def scan_devices(timeout: float = 5.0) -> list[dict]:
        """扫描附近 BLE 设备，返回 [{name, address}] 列表（独立调用，非线程内使用）。"""
        if not BLEAK_AVAILABLE:
            logger.error("bleak 未安装")
            return []
        devices = await BleakScanner.discover(timeout=timeout)
        result = [{"name": d.name or "Unknown", "address": d.address} for d in devices]
        for d in result:
            logger.info("BLE 设备: %s  %s", d["address"], d["name"])
        return result
