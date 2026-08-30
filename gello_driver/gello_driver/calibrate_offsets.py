"""
calibrate_offsets.py — GELLO baseline joint-offset calibration.

Place the GELLO leader arm in its usual "parked" starting pose, then run
this script.  It reads the raw Dynamixel encoder values once and writes

    joint_offsets_deg = round(raw_deg / snap_deg) * snap_deg

back to gello_params.yaml.  The result is a stable, always-consistent
baseline: every subsequent run starts with GELLO output within ±snap/2°
of zero on each joint.  Add per-joint corrections manually on top of this
baseline to align with the real robot's home pose.

Run
---
    ros2 run gello_driver calibrate_offsets
    ros2 run gello_driver calibrate_offsets --config /path/to/gello_params.yaml
    ros2 run gello_driver calibrate_offsets --dry-run
    ros2 run gello_driver calibrate_offsets --snap-deg 45
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory

from gello_driver.dynamixel_driver import DynamixelDriver


PARAMS_KEY_PATH = ("gello_node", "ros__parameters")


def _load_params(config_path: Path) -> dict:
    with config_path.open("r") as f:
        doc = yaml.safe_load(f)
    node = doc
    for key in PARAMS_KEY_PATH:
        if key not in node:
            raise KeyError(f"'{key}' missing in {config_path}")
        node = node[key]
    return node


def _snap_offsets_deg(raw_rad: np.ndarray, snap_deg: float) -> np.ndarray:
    """Snap each raw reading (rad) to the nearest multiple of snap_deg (°)."""
    raw_deg = np.degrees(raw_rad)
    return np.round(raw_deg / snap_deg) * snap_deg


def _rewrite_offsets_line(config_path: Path, new_offsets_deg: List[float]) -> None:
    """
    Replace the `joint_offsets: [...]` line (single-line list form) in the
    YAML while preserving everything else, including comments.
    """
    text = config_path.read_text()
    formatted = ", ".join(f"{v:.1f}" for v in new_offsets_deg)
    pattern = re.compile(r"^(\s*joint_offsets\s*:\s*)\[[^\]]*\]", re.MULTILINE)
    new_text, n = pattern.subn(lambda m: f"{m.group(1)}[{formatted}]", text)
    if n == 0:
        raise RuntimeError(
            "Could not find single-line `joint_offsets: [...]` in "
            f"{config_path}; refusing to guess."
        )
    config_path.write_text(new_text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="GELLO baseline offset calibration.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to gello_params.yaml (default: installed package share).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print computed offsets but do not modify the YAML.",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.5,
        help="Seconds to let the reader thread populate positions.",
    )
    parser.add_argument(
        "--snap-deg",
        type=float,
        default=90.0,
        help="Snap computed offsets to the nearest multiple of this angle "
             "(default 90°). Use 45 if your GELLO mounts on a finer grid.",
    )
    args = parser.parse_args(argv)

    if args.config is None:
        share = Path(get_package_share_directory("gello_driver"))
        args.config = share / "config" / "gello_params.yaml"

    if not args.config.exists():
        print(f"[calibrate] Config not found: {args.config}", file=sys.stderr)
        return 1

    params = _load_params(args.config)
    port = params["port"]
    baudrate = int(params["baudrate"])
    joint_ids = list(params["joint_ids"])
    signs = np.array([int(s) for s in params["joint_signs"]], dtype=int)
    prev_offsets_deg = np.array(params["joint_offsets"], dtype=float)

    if not (len(joint_ids) == len(signs) == len(prev_offsets_deg)):
        print(
            "[calibrate] length mismatch across joint_ids / joint_signs / "
            "joint_offsets.",
            file=sys.stderr,
        )
        return 1

    print(f"[calibrate] Config: {args.config}")
    print(f"[calibrate] Port:   {port} @ {baudrate}")
    print(f"[calibrate] IDs:    {joint_ids}")
    print(f"[calibrate] Snap:   {args.snap_deg}°")
    print()
    input(
        "Place GELLO in its usual parked/start pose, then press Enter to sample..."
    )

    gripper_id = int(params.get("gripper_id", -1))
    all_ids = list(joint_ids) + ([gripper_id] if gripper_id >= 0 else [])

    driver = DynamixelDriver(
        ids=all_ids,
        port=port,
        baudrate=baudrate,
        use_fake_fallback=False,
    )
    try:
        time.sleep(args.settle_seconds)
        raw_all = driver.get_joints()
    finally:
        driver.close()

    raw_rad = np.asarray(raw_all[: len(joint_ids)], dtype=float)
    new_offsets_deg = _snap_offsets_deg(raw_rad, args.snap_deg)
    new_offsets_rad = np.radians(new_offsets_deg)

    baseline_output = (raw_rad - new_offsets_rad) * signs

    print()
    print(f"[calibrate] Raw (°):          {np.degrees(raw_rad).tolist()}")
    print(f"[calibrate] Old offsets (°):  {prev_offsets_deg.tolist()}")
    print(f"[calibrate] New offsets (°):  {new_offsets_deg.tolist()}")
    print(f"[calibrate] Baseline output:  {baseline_output.tolist()}  (rad)")
    print("[calibrate] --> Add per-joint corrections to joint_offsets manually "
          "to reach the robot's home pose.")

    if args.dry_run:
        print("[calibrate] --dry-run: YAML not modified.")
        return 0

    _rewrite_offsets_line(args.config, new_offsets_deg.tolist())
    print(f"[calibrate] Updated joint_offsets in {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
