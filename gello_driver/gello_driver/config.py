"""
config.py — GELLO configuration dataclass and PORT_CONFIG_MAP.

Add your device's serial port (by-id path) here with the matching
joint calibration values.  The gello_node.py will automatically pick
up the correct config when it detects a known port.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class GelloConfig:
    """Full calibration configuration for one GELLO device."""

    # ------------------------------------------------------------------ #
    # Dynamixel servo IDs (arm joints only, NOT including gripper)
    # ------------------------------------------------------------------ #
    joint_ids: List[int]

    # ------------------------------------------------------------------ #
    # Per-joint calibration
    # ------------------------------------------------------------------ #
    joint_offsets: List[float]
    """Offset added to the raw Dynamixel reading (radians).
    Should be a multiple of π/2 so that the home pose is predictable."""

    joint_signs: List[int]
    """Sign correction per joint: +1 or -1.
    Accounts for the physical mounting direction of each servo."""

    # ------------------------------------------------------------------ #
    # Gripper (optional)
    # ------------------------------------------------------------------ #
    gripper_id: Optional[int] = None
    """Dynamixel ID of the gripper servo.  None = no gripper."""

    gripper_open_pos_deg: float = 0.0
    """Dynamixel position (degrees) that corresponds to fully open."""

    gripper_closed_pos_deg: float = 0.0
    """Dynamixel position (degrees) that corresponds to fully closed."""

    # ------------------------------------------------------------------ #
    # Communication
    # ------------------------------------------------------------------ #
    baudrate: int = 57600

    # ------------------------------------------------------------------ #
    # Smoothing
    # ------------------------------------------------------------------ #
    alpha: float = 0.5
    """Exponential smoothing factor for joint readings (0–1).
    1.0 = no smoothing (raw), 0.0 = frozen."""

    def __post_init__(self) -> None:
        assert len(self.joint_ids) == len(self.joint_offsets), (
            f"joint_ids length ({len(self.joint_ids)}) must match "
            f"joint_offsets length ({len(self.joint_offsets)})"
        )
        assert len(self.joint_ids) == len(self.joint_signs), (
            f"joint_ids length ({len(self.joint_ids)}) must match "
            f"joint_signs length ({len(self.joint_signs)})"
        )
        for s in self.joint_signs:
            assert s in (1, -1), f"joint_signs must be +1 or -1, got {s}"

    @property
    def all_ids(self) -> List[int]:
        """All servo IDs including gripper (if present)."""
        ids = list(self.joint_ids)
        if self.gripper_id is not None:
            ids.append(self.gripper_id)
        return ids

    @property
    def n_joints(self) -> int:
        return len(self.joint_ids)

    @property
    def gripper_range_rad(self):
        """(open_rad, closed_rad) or None."""
        if self.gripper_id is None:
            return None
        return (
            math.radians(self.gripper_open_pos_deg),
            math.radians(self.gripper_closed_pos_deg),
        )


# --------------------------------------------------------------------------- #
# PORT_CONFIG_MAP
# --------------------------------------------------------------------------- #
# KEY  : The device path.  Use /dev/serial/by-id/<device> so it is stable
#        across reboots and USB port changes.
# VALUE: A GelloConfig with calibrated offsets and signs for your robot.
# --------------------------------------------------------------------------- #

_π = math.pi

PORT_CONFIG_MAP: Dict[str, GelloConfig] = {

    # ---------------------------------------------------------------------- #
    # Franka FR3 / Panda  (7-DOF)
    # Replace the key with your actual by-id path.
    # ---------------------------------------------------------------------- #
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT3M9NVB-if00-port0": GelloConfig(
        joint_ids=[1, 2, 3, 4, 5, 6, 7],
        joint_offsets=[
            3 * _π / 2,
            2 * _π / 2,
            1 * _π / 2,
            4 * _π / 2,
            -2 * _π / 2 + 2 * _π,
            3 * _π / 2,
            4 * _π / 2,
        ],
        joint_signs=[1, -1, 1, 1, 1, -1, 1],
        gripper_id=8,
        gripper_open_pos_deg=195.0,
        gripper_closed_pos_deg=152.0,
        baudrate=57600,
        alpha=0.5,
    ),

    # ---------------------------------------------------------------------- #
    # Example: 6-DOF GELLO for UR / generic arm
    # ---------------------------------------------------------------------- #
    # "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7WBEIA-if00-port0": GelloConfig(
    #     joint_ids=[1, 2, 3, 4, 5, 6],
    #     joint_offsets=[
    #         0,
    #         1 * _π / 2 + _π,
    #         _π / 2,
    #         _π / 2,
    #         _π - 2 * _π / 2,
    #         -1 * _π / 2 + 2 * _π,
    #     ],
    #     joint_signs=[1, 1, -1, 1, 1, 1],
    #     gripper_id=7,
    #     gripper_open_pos_deg=20.0,
    #     gripper_closed_pos_deg=-22.0,
    #     baudrate=57600,
    # ),
}
