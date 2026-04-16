"""
franka_env.py — MuJoCo environment wrapper for Franka Emika FR3 / Panda.

Handles:
- Downloading / locating the MuJoCo XML model (mujoco_menagerie)
- Loading the model and creating a viewer-friendly data structure
- Applying joint position commands and stepping the physics
"""

from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Model acquisition
# --------------------------------------------------------------------------- #

# We use the Franka Panda model bundled with mujoco_menagerie (Apache 2.0).
# If you have a local copy, point FRANKA_XML_PATH env-var at it.
_MENAGERIE_BASE = (
    "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/"
)

# Files that need to be downloaded (relative to the franka_fr3 sub-folder)
_FR3_FILES = [
    "franka_fr3/fr3.xml",
    "franka_fr3/fr3_nohand.xml",
    "franka_fr3/assets/link0.stl",
    "franka_fr3/assets/link1.stl",
    "franka_fr3/assets/link2.stl",
    "franka_fr3/assets/link3.stl",
    "franka_fr3/assets/link4.stl",
    "franka_fr3/assets/link5.stl",
    "franka_fr3/assets/link6.stl",
    "franka_fr3/assets/link7.stl",
    "franka_fr3/assets/hand.stl",
    "franka_fr3/assets/finger.stl",
]

# Joint names as they appear in the MuJoCo XML for FR3
FRANKA_JOINT_NAMES_MJ = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]


def _get_assets_dir() -> Path:
    """Return (and create if needed) the directory where model files live."""
    env_path = os.environ.get("FRANKA_XML_PATH")
    if env_path:
        return Path(env_path).expanduser()
    # Default: next to this file
    return Path(__file__).parent.parent / "assets" / "franka_fr3"


def ensure_franka_model(assets_dir: Optional[Path] = None) -> Path:
    """
    Ensure the Franka FR3 MuJoCo XML is available locally.

    Returns the path to ``fr3.xml``.
    """
    assets_dir = assets_dir or _get_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)
    xml_path = assets_dir / "fr3.xml"

    if xml_path.exists():
        return xml_path

    print("[franka_env] Downloading Franka FR3 model from mujoco_menagerie ...")
    (assets_dir / "assets").mkdir(exist_ok=True)

    for rel in _FR3_FILES:
        subdir = Path(rel).parent.name   # "" or "assets"
        fname = Path(rel).name
        dest = assets_dir / (Path(rel).relative_to("franka_fr3"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = _MENAGERIE_BASE + rel
        try:
            print(f"  GET {url}")
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(
                f"[franka_env] WARNING: Could not download {url}: {exc}\n"
                f"  Set FRANKA_XML_PATH to a local copy of the XML."
            )

    return xml_path


# --------------------------------------------------------------------------- #
# Environment class
# --------------------------------------------------------------------------- #

class FrankaEnv:
    """
    Thin wrapper around a MuJoCo Franka FR3 model.

    Parameters
    ----------
    xml_path:
        Path to the `fr3.xml` file.  If None, auto-download is attempted.
    joint_names:
        MuJoCo joint names to control (must match XML).
    render:
        Whether to open an interactive viewer (requires a display).
    """

    def __init__(
        self,
        xml_path: Optional[Path] = None,
        joint_names: Optional[List[str]] = None,
        render: bool = True,
    ) -> None:
        try:
            import mujoco
            import mujoco.viewer as mj_viewer
        except ImportError as exc:
            raise ImportError(
                "mujoco is not installed.  Run: pip install mujoco"
            ) from exc

        self._mj = mujoco
        self._mj_viewer = mj_viewer

        xml_path = xml_path or ensure_franka_model()
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)

        self._joint_names = joint_names or FRANKA_JOINT_NAMES_MJ
        self._joint_ids: List[int] = [
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            for jn in self._joint_names
        ]
        for jid, jname in zip(self._joint_ids, self._joint_names):
            if jid < 0:
                raise RuntimeError(
                    f"Joint '{jname}' not found in MuJoCo model.  "
                    f"Check the XML file at {xml_path}."
                )

        # Get corresponding qpos indices
        self._qpos_idxs: List[int] = [
            self._model.jnt_qposadr[jid] for jid in self._joint_ids
        ]

        self._render = render
        self._viewer = None
        self._target_qpos = np.zeros(len(self._joint_names))

        # Initialise to current (default) pose
        mujoco.mj_forward(self._model, self._data)

    # ----------------------------------------------------------------------- #
    # Viewer management
    # ----------------------------------------------------------------------- #

    def launch_viewer(self) -> None:
        """Launch the passive MuJoCo viewer (call once from main thread)."""
        if not self._render:
            return
        # passive viewer: physics is stepped externally
        self._viewer = self._mj_viewer.launch_passive(
            self._model, self._data,
            show_left_ui=False,
            show_right_ui=False,
        )
        print("[FrankaEnv] MuJoCo viewer launched.")

    def viewer_is_running(self) -> bool:
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    # ----------------------------------------------------------------------- #
    # Command interface
    # ----------------------------------------------------------------------- #

    def set_joint_position_target(self, positions: np.ndarray) -> None:
        """Set desired joint positions (radians). Physics will track these."""
        if len(positions) != len(self._joint_names):
            raise ValueError(
                f"Expected {len(self._joint_names)} positions, "
                f"got {len(positions)}"
            )
        self._target_qpos = np.array(positions, dtype=float)

    # ----------------------------------------------------------------------- #
    # Stepping
    # ----------------------------------------------------------------------- #

    def step(self) -> None:
        """
        Apply the position target as a PD setpoint and step physics.

        We use the built-in equality / position actuators when available,
        otherwise fall back to direct qpos injection (kinematic).
        """
        # Simple position servo: directly set qpos and integrate one step
        # For a more realistic simulation you can swap this with torque control.
        for idx, qpos_addr in enumerate(self._qpos_idxs):
            self._data.qpos[qpos_addr] = self._target_qpos[idx]
            # Zero velocities to avoid drift
            # (comment out for dynamics-based control)
            qadr_vel = self._model.jnt_dofadr[self._joint_ids[idx]]
            self._data.qvel[qadr_vel] = 0.0

        self._mj.mj_forward(self._model, self._data)
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    # ----------------------------------------------------------------------- #
    # Cleanup
    # ----------------------------------------------------------------------- #

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
