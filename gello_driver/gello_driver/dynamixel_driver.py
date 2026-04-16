"""
dynamixel_driver.py — Thread-safe Dynamixel SDK wrapper.

Continuously reads joint positions (and optionally velocities) from
Dynamixel servos in a background thread, providing low-latency access
via get_joints().

Supports a FakeDriver fallback so the pipeline can be exercised
without real hardware.
"""

from __future__ import annotations

import os
import subprocess
import time
from threading import Event, Lock, Thread
from typing import List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Dynamixel control-table addresses (X-series, Protocol 2.0)
# --------------------------------------------------------------------------- #
ADDR_OPERATING_MODE      = 11
ADDR_CURRENT_LIMIT       = 38   # 2 bytes, EEPROM — write only when torque OFF
LEN_CURRENT_LIMIT        = 2    # unit: depends on servo model (see docs below)
ADDR_TORQUE_ENABLE       = 64
ADDR_LED                 = 65   # 1 byte: 1 = on, 0 = off
ADDR_GOAL_CURRENT        = 102
LEN_GOAL_CURRENT         = 2
ADDR_PRESENT_VELOCITY    = 128
LEN_PRESENT_VELOCITY     = 4
ADDR_PRESENT_POSITION    = 132
LEN_PRESENT_POSITION     = 4
ADDR_GOAL_POSITION       = 116
LEN_GOAL_POSITION        = 4

TORQUE_ENABLE            = 1
TORQUE_DISABLE           = 0
POSITION_CONTROL_MODE    = 3
CURRENT_CONTROL_MODE     = 0

# ---------------------------------------------------------------------------
# Current unit per servo model (mA per raw unit)
# ---------------------------------------------------------------------------
# XC330-T288-T  : 1.0  mA / unit  (max 1193 units / ~1193 mA)
# XC330-T181-T  : 1.0  mA / unit
# XM430-W210-T  : 2.69 mA / unit  (max 1263 mA  ≈  469 units)
# XM540-W150-T  : 2.69 mA / unit
# XH430-V210-R  : 1.0  mA / unit
# Use CURRENT_UNIT_MA to convert mA → raw unit when calling set_current_limit.
CURRENT_UNIT_mA: dict = {
    "XC330":  1.0,
    "XM430":  2.69,
    "XM540":  2.69,
    "XH430":  1.0,
}


# --------------------------------------------------------------------------- #
# Fake driver
# --------------------------------------------------------------------------- #

class FakeDynamixelDriver:
    """Drop-in replacement when no hardware is connected."""

    def __init__(self, ids: Sequence[int]) -> None:
        self._ids = list(ids)
        self._positions = np.zeros(len(ids), dtype=float)
        self._velocities = np.zeros(len(ids), dtype=float)
        self._torque_enabled = False

    # --- properties -------------------------------------------------------- #
    @property
    def is_fake(self) -> bool:
        return True

    # --- public API -------------------------------------------------------- #
    def set_torque_mode(self, enable: bool) -> None:
        self._torque_enabled = enable

    def set_torque_mode_ids(self, ids: Sequence[int], enable: bool) -> None:
        """Set torque enable for specific servo IDs (fake: no-op)."""
        pass

    def set_operating_mode_ids(self, ids: Sequence[int], mode: int) -> None:
        """Set operating mode for specific servo IDs (fake: no-op)."""
        pass

    def set_led(self, enable: bool, ids: Optional[Sequence[int]] = None) -> None:
        """Turn LED on/off (fake: no-op)."""
        pass

    def set_current_limit_single(self, dxl_id: int, current_ma: float,
                                  current_unit_ma: float = 1.0) -> None:
        """
        Limit the maximum current (= torque) for a single servo.
        Torque must be DISABLED before calling (EEPROM register).
        Fake driver: no-op.
        """
        pass

    def set_goal_position_single(self, dxl_id: int, position_rad: float) -> None:
        """Set goal position for a single servo in radians (fake: no-op)."""
        pass

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def get_joints(self) -> np.ndarray:
        return self._positions.copy()

    def get_positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._positions.copy(), self._velocities.copy()

    def set_joints(self, joint_angles: Sequence[float]) -> None:
        self._positions = np.array(joint_angles, dtype=float)

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Real driver
# --------------------------------------------------------------------------- #

class DynamixelDriver:
    """
    Thread-safe Dynamixel driver using Protocol 2.0 GroupSyncRead/Write.

    A background thread continuously reads position (and velocity) so that
    get_joints() always returns the freshest value without blocking the
    calling thread on serial I/O.

    Parameters
    ----------
    ids:
        Servo IDs to manage.
    port:
        Serial device path, e.g. ``/dev/ttyUSB0`` or a by-id path.
    baudrate:
        Dynamixel bus baudrate.
    max_retries:
        How many times to attempt port initialisation before giving up.
    use_fake_fallback:
        If True, fall back to FakeDynamixelDriver on failure instead of
        raising an exception.
    """

    def __init__(
        self,
        ids: Sequence[int],
        port: str = "/dev/ttyUSB0",
        baudrate: int = 57600,
        max_retries: int = 3,
        use_fake_fallback: bool = True,
    ) -> None:
        try:
            from dynamixel_sdk import (
                GroupSyncRead,
                GroupSyncWrite,
                PacketHandler,
                PortHandler,
                COMM_SUCCESS,
                DXL_HIBYTE,
                DXL_HIWORD,
                DXL_LOBYTE,
                DXL_LOWORD,
            )
            self._dxl = {
                "GroupSyncRead": GroupSyncRead,
                "GroupSyncWrite": GroupSyncWrite,
                "PacketHandler": PacketHandler,
                "PortHandler": PortHandler,
                "COMM_SUCCESS": COMM_SUCCESS,
                "DXL_HIBYTE": DXL_HIBYTE,
                "DXL_HIWORD": DXL_HIWORD,
                "DXL_LOBYTE": DXL_LOBYTE,
                "DXL_LOWORD": DXL_LOWORD,
            }
        except ImportError as e:
            raise ImportError(
                "dynamixel_sdk is not installed.  Run: pip install dynamixel-sdk"
            ) from e

        self._ids = list(ids)
        self._port = port
        self._baudrate = baudrate
        self._max_retries = max_retries
        self._use_fake_fallback = use_fake_fallback

        self._lock = Lock()
        self._stop_event = Event()
        self._torque_enabled = False
        self._is_fake = False

        # Shared state updated by the reader thread
        self._positions: Optional[np.ndarray] = None
        self._velocities: Optional[np.ndarray] = None

        if not self._init_with_retries():
            if use_fake_fallback:
                print(
                    f"[DynamixelDriver] Could not connect to {port}. "
                    "Using FakeDynamixelDriver."
                )
                self._is_fake = True
                self._positions = np.zeros(len(ids), dtype=float)
                self._velocities = np.zeros(len(ids), dtype=float)
            else:
                raise RuntimeError(
                    f"[DynamixelDriver] Failed to initialise after "
                    f"{max_retries} attempts."
                )

    # ----------------------------------------------------------------------- #
    # Internal initialisation helpers
    # ----------------------------------------------------------------------- #

    def _init_with_retries(self) -> bool:
        for attempt in range(self._max_retries):
            print(
                f"[DynamixelDriver] Init attempt {attempt + 1}/{self._max_retries} "
                f"on {self._port} ..."
            )
            self._fix_port_permissions()
            try:
                self._init_hardware()
                print(f"[DynamixelDriver] Connected to {self._port}")
                return True
            except Exception as exc:
                print(f"[DynamixelDriver] Attempt failed: {exc}")
                if attempt < self._max_retries - 1:
                    time.sleep(2.0)
        return False

    def _init_hardware(self) -> None:
        dxl = self._dxl
        self._port_handler = dxl["PortHandler"](self._port)
        self._packet_handler = dxl["PacketHandler"](2.0)

        # GroupSyncRead: read velocity + position in one transaction
        self._sync_read = dxl["GroupSyncRead"](
            self._port_handler,
            self._packet_handler,
            ADDR_PRESENT_VELOCITY,
            LEN_PRESENT_VELOCITY + LEN_PRESENT_POSITION,
        )
        # GroupSyncWrite: write goal position
        self._sync_write = dxl["GroupSyncWrite"](
            self._port_handler,
            self._packet_handler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        if not self._port_handler.openPort():
            raise RuntimeError(f"Cannot open port {self._port}")
        if not self._port_handler.setBaudRate(self._baudrate):
            raise RuntimeError(f"Cannot set baudrate {self._baudrate}")

        for dxl_id in self._ids:
            if not self._sync_read.addParam(dxl_id):
                raise RuntimeError(f"addParam failed for ID {dxl_id}")

        # Disable torque on startup (GELLO is a leader — free to move)
        self.set_torque_mode(False)

        # Start background reader
        self._reader_thread = Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    # ----------------------------------------------------------------------- #
    # Background position reader
    # ----------------------------------------------------------------------- #

    def _reader_loop(self) -> None:
        COMM_SUCCESS = self._dxl["COMM_SUCCESS"]
        while not self._stop_event.is_set():
            time.sleep(0.001)  # ~1 kHz attempt rate
            with self._lock:
                result = self._sync_read.txRxPacket()
                if result != COMM_SUCCESS:
                    continue

                positions = np.zeros(len(self._ids), dtype=int)
                velocities = np.zeros(len(self._ids), dtype=int)

                ok = True
                for i, dxl_id in enumerate(self._ids):
                    if self._sync_read.isAvailable(
                        dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                    ):
                        v = self._sync_read.getData(
                            dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY
                        )
                        if v > 0x7FFFFFFF:
                            v -= 0x100000000
                        velocities[i] = v
                    else:
                        ok = False
                        break

                    if self._sync_read.isAvailable(
                        dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                    ):
                        p = self._sync_read.getData(
                            dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                        )
                        if p > 0x7FFFFFFF:
                            p -= 0x100000000
                        positions[i] = p
                    else:
                        ok = False
                        break

                if ok:
                    self._positions = positions
                    self._velocities = velocities

    # ----------------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------------- #

    @property
    def is_fake(self) -> bool:
        return self._is_fake

    # --- torque ------------------------------------------------------------ #
    def set_torque_mode(self, enable: bool) -> None:
        """Set torque enable for ALL managed servo IDs."""
        if self._is_fake:
            self._torque_enabled = enable
            return
        self.set_torque_mode_ids(self._ids, enable)
        self._torque_enabled = enable

    def set_torque_mode_ids(self, ids: Sequence[int], enable: bool) -> None:
        """Set torque enable for a specific subset of servo IDs."""
        if self._is_fake:
            return
        COMM_SUCCESS = self._dxl["COMM_SUCCESS"]
        val = TORQUE_ENABLE if enable else TORQUE_DISABLE
        with self._lock:
            for dxl_id in ids:
                res, err = self._packet_handler.write1ByteTxRx(
                    self._port_handler, dxl_id, ADDR_TORQUE_ENABLE, val
                )
                if res != COMM_SUCCESS or err != 0:
                    raise RuntimeError(
                        f"set_torque_mode_ids failed for ID {dxl_id} "
                        f"(res={res}, err={err})"
                    )

    def set_operating_mode_ids(self, ids: Sequence[int], mode: int) -> None:
        """
        Set operating mode for a subset of servo IDs.
        IMPORTANT: torque must be DISABLED before calling this.
        """
        if self._is_fake:
            return
        COMM_SUCCESS = self._dxl["COMM_SUCCESS"]
        with self._lock:
            for dxl_id in ids:
                res, err = self._packet_handler.write1ByteTxRx(
                    self._port_handler, dxl_id, ADDR_OPERATING_MODE, mode
                )
                if res != COMM_SUCCESS or err != 0:
                    raise RuntimeError(
                        f"set_operating_mode_ids failed for ID {dxl_id} "
                        f"(res={res}, err={err})"
                    )

    # --- LED --------------------------------------------------------------- #
    def set_led(self, enable: bool, ids: Optional[Sequence[int]] = None) -> None:
        """
        Turn LED on (True) or off (False) for the given IDs.
        If ids is None, applies to all managed IDs.
        LED can be set regardless of torque state.
        """
        if self._is_fake:
            return
        COMM_SUCCESS = self._dxl["COMM_SUCCESS"]
        target_ids = self._ids if ids is None else list(ids)
        val = 1 if enable else 0
        with self._lock:
            for dxl_id in target_ids:
                res, err = self._packet_handler.write1ByteTxRx(
                    self._port_handler, dxl_id, ADDR_LED, val
                )
                if res != COMM_SUCCESS or err != 0:
                    # Non-fatal: log but don't raise
                    print(
                        f"[DynamixelDriver] set_led warning for ID {dxl_id}: "
                        f"res={res}, err={err}"
                    )

    # --- current limit (torque cap) ---------------------------------------- #
    def set_current_limit_single(
        self,
        dxl_id: int,
        current_ma: float,
        current_unit_ma: float = 1.0,
    ) -> None:
        """
        Write the Current Limit register (addr 38) for a single servo.

        This caps the maximum current output in any control mode, effectively
        limiting the maximum torque the servo can exert.

        IMPORTANT: This is an EEPROM register — torque must be DISABLED
        before writing.  Call this before set_torque_mode_ids().

        Parameters
        ----------
        dxl_id : int
            Dynamixel servo ID.
        current_ma : float
            Desired maximum current in milliamperes (mA).
        current_unit_ma : float
            mA per raw unit for this servo model.
            e.g. XC330 → 1.0,  XM430 → 2.69  (see CURRENT_UNIT_mA dict)
        """
        if self._is_fake:
            return
        raw = max(0, int(current_ma / current_unit_ma))
        COMM_SUCCESS = self._dxl["COMM_SUCCESS"]
        with self._lock:
            res, err = self._packet_handler.write2ByteTxRx(
                self._port_handler, dxl_id, ADDR_CURRENT_LIMIT, raw
            )
        if res != COMM_SUCCESS or err != 0:
            raise RuntimeError(
                f"set_current_limit_single failed for ID {dxl_id} "
                f"(res={res}, err={err}, raw={raw})"
            )

    # --- single-servo goal position ---------------------------------------- #
    def set_goal_position_single(self, dxl_id: int, position_rad: float) -> None:
        """
        Write a goal position (radians) to a single servo.
        Torque must be enabled for that servo beforehand.
        """
        if self._is_fake:
            return
        dxl = self._dxl
        val = int(position_rad * 2048.0 / np.pi)
        with self._lock:
            res, err = self._packet_handler.write4ByteTxRx(
                self._port_handler, dxl_id, ADDR_GOAL_POSITION, val
            )
        if res != dxl["COMM_SUCCESS"] or err != 0:
            raise RuntimeError(
                f"set_goal_position_single failed for ID {dxl_id} "
                f"(res={res}, err={err})"
            )


    def torque_enabled(self) -> bool:
        return self._torque_enabled

    # --- reading ----------------------------------------------------------- #
    def get_joints(self) -> np.ndarray:
        """Return current joint positions in radians."""
        if self._is_fake:
            return (self._positions or np.zeros(len(self._ids))).copy()
        while self._positions is None:
            time.sleep(0.01)
        return self._positions.copy() / 2048.0 * np.pi

    def get_positions_and_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (positions_rad, velocities_rad_s)."""
        if self._is_fake:
            p = (self._positions or np.zeros(len(self._ids))).copy()
            v = (self._velocities or np.zeros(len(self._ids))).copy()
            return p, v
        while self._positions is None or self._velocities is None:
            time.sleep(0.01)
        pos_rad = self._positions.copy() / 2048.0 * np.pi
        vel_rads = self._velocities.copy() * 0.229 * 2.0 * np.pi / 60.0
        return pos_rad, vel_rads

    # --- writing (optional — GELLO is leader, so rarely used here) --------- #
    def set_joints(self, joint_angles: Sequence[float]) -> None:
        if self._is_fake:
            self._positions = np.array(joint_angles, dtype=float)
            return
        if not self._torque_enabled:
            raise RuntimeError("Enable torque before commanding joint positions.")
        dxl = self._dxl
        with self._lock:
            for dxl_id, angle in zip(self._ids, joint_angles):
                val = int(angle * 2048.0 / np.pi)
                param = [
                    dxl["DXL_LOBYTE"](dxl["DXL_LOWORD"](val)),
                    dxl["DXL_HIBYTE"](dxl["DXL_LOWORD"](val)),
                    dxl["DXL_LOBYTE"](dxl["DXL_HIWORD"](val)),
                    dxl["DXL_HIBYTE"](dxl["DXL_HIWORD"](val)),
                ]
                if not self._sync_write.addParam(dxl_id, param):
                    raise RuntimeError(f"addParam failed for ID {dxl_id}")
            res = self._sync_write.txPacket()
            if res != dxl["COMM_SUCCESS"]:
                raise RuntimeError("SyncWrite txPacket failed")
            self._sync_write.clearParam()

    # --- cleanup ----------------------------------------------------------- #
    def close(self) -> None:
        if self._is_fake:
            return
        self._stop_event.set()
        self._reader_thread.join(timeout=2.0)
        self._port_handler.closePort()

    # ----------------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------------- #

    def _fix_port_permissions(self) -> None:
        if not os.path.exists(self._port):
            return
        try:
            subprocess.run(
                ["sudo", "chmod", "a+rw", self._port],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass
