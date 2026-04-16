import os
from pathlib import Path
from typing import Optional, List, Union

import mujoco
import numpy as np

def _get_assets_dir() -> Path:
    """Return the directory where model files live (installed share directory)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('franka_mujoco_receiver')) / "assets" / "franka_fr3"
    except Exception:
        # Fallback to local path for development/non-ROS testing
        return Path(__file__).parent.parent / "assets" / "franka_fr3"

def ensure_franka_model() -> Path:
    """Returns the path to the internal fr3.xml."""
    assets_dir = _get_assets_dir()
    xml_path = assets_dir / "fr3.xml"
    if not xml_path.exists():
        raise FileNotFoundError(f"Franka model not found at {xml_path}. Ensure assets are copied.")
    return xml_path

class FrankaEnv:
    def __init__(
        self,
        xml_path: Optional[Union[str, Path]] = None,
        joint_names: Optional[List[str]] = None,
        render: bool = True,
    ):
        xml_path = xml_path or ensure_franka_model()
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        
        # We always want the newest fr3_ joints
        if joint_names is None:
            joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
        
        self._joint_names = joint_names
        self._joint_ids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
        
        # Verify IDs
        for name, jid in zip(self._joint_names, self._joint_ids):
            if jid == -1:
                raise ValueError(f"Joint {name} not found in MuJoCo model.")

        self._render = render
        self._viewer = None
        self._target_qpos = np.zeros(len(self._joint_names))
        mujoco.mj_forward(self._model, self._data)

    def set_joint_position_target(self, positions: np.ndarray) -> None:
        if len(positions) != len(self._joint_ids):
            raise ValueError(f"Expected {len(self._joint_ids)} positions, got {len(positions)}")
        self._target_qpos = np.array(positions, dtype=float)

    def step(self) -> None:
        """Advance simulation by exactly one model timestep."""
        # Control via actuators (ctrl) using latest target
        for i in range(len(self._target_qpos)):
            if i < self._model.nu:
                self._data.ctrl[i] = self._target_qpos[i]

        mujoco.mj_step(self._model, self._data)
        
    def sync_viewer(self) -> None:
        """Sync the viewer if active."""
        if self._render and self._viewer is not None:
            self._viewer.sync()

    def _display(self) -> None:
        if self._viewer is None:
            from mujoco import viewer
            self._viewer = viewer.launch_passive(self._model, self._data)
        self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()

    def get_joint_positions(self) -> np.ndarray:
        return np.array([self._data.qpos[self._model.jnt_qposadr[jid]] for jid in self._joint_ids])
