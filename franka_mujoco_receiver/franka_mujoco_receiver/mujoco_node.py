"""
mujoco_node.py — ROS 2 node: receives /gello/joint_command and drives
a Franka robot in a MuJoCo viewer.

Architecture
------------
- The MuJoCo viewer MUST run on the main thread.
- ROS 2 spin runs in a separate thread.
- Joint commands received by the subscriber are stored in a shared
  numpy array protected by a threading.Lock and consumed by the main
  loop at the viewer / physics timestep.

Usage
-----
ros2 launch franka_mujoco_receiver mujoco_receiver.launch.py
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from franka_mujoco_receiver.franka_env import FrankaEnv, FRANKA_JOINT_NAMES_MJ


# Mapping from incoming /gello/joint_command name to MuJoCo joint name
# (strip the "fr3_" prefix that Franka ROS 2 uses)
def _ros_to_mj_name(ros_name: str) -> str:
    return ros_name.replace("fr3_", "")


class FrankaMujocoNode(Node):
    """
    ROS 2 node that bridges /gello/joint_command → MuJoCo Franka simulation.

    Parameters
    ----------
    xml_path        : str   — path to fr3.xml (empty = auto-download)
    joint_names_mj  : list  — MuJoCo joint names to control
    step_rate       : float — physics / viewer update rate (Hz)
    """

    def __init__(self) -> None:
        super().__init__("franka_mujoco_receiver")

        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("xml_path", "")
        self.declare_parameter(
            "joint_names_mj",
            ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
        )
        self.declare_parameter("step_rate", 50.0)

        xml_path_str = (
            self.get_parameter("xml_path").get_parameter_value().string_value
        )
        self._xml_path: Optional[Path] = (
            Path(xml_path_str) if xml_path_str else None
        )
        self._joint_names_mj: List[str] = list(
            self.get_parameter("joint_names_mj")
            .get_parameter_value()
            .string_array_value
        )
        self._step_rate: float = (
            self.get_parameter("step_rate").get_parameter_value().double_value
        )

        # ------------------------------------------------------------------ #
        # Shared state (producer: ROS callback, consumer: main loop)
        # ------------------------------------------------------------------ #
        self._lock = threading.Lock()
        self._target: Optional[np.ndarray] = None  # None until first command
        self._n_joints = len(self._joint_names_mj)

        # ------------------------------------------------------------------ #
        # Subscriber
        # ------------------------------------------------------------------ #
        self._sub = self.create_subscription(
            JointState,
            "/gello/joint_command",
            self._joint_command_cb,
            10,
        )
        self.get_logger().info(
            "FrankaMujocoNode started — listening on /gello/joint_command"
        )

    # ----------------------------------------------------------------------- #
    # Subscriber callback
    # ----------------------------------------------------------------------- #

    def _joint_command_cb(self, msg: JointState) -> None:
        """
        Receive a JointState command and extract positions for our joints.

        The message may contain extra joints (e.g. gripper); we only pick
        the ones whose names map to our MuJoCo joint list.
        """
        if len(msg.name) == 0 or len(msg.position) == 0:
            return

        # Build a name → position mapping from the incoming message
        name_to_pos = {
            name: pos for name, pos in zip(msg.name, msg.position)
        }

        positions = np.zeros(self._n_joints)
        for i, mj_name in enumerate(self._joint_names_mj):
            # Accept both "joint1" (MuJoCo style) and "fr3_joint1" (ROS style)
            ros_name = f"fr3_{mj_name}"
            if mj_name in name_to_pos:
                positions[i] = name_to_pos[mj_name]
            elif ros_name in name_to_pos:
                positions[i] = name_to_pos[ros_name]
            # If neither found, leave as zero (safe default)

        with self._lock:
            self._target = positions

    # ----------------------------------------------------------------------- #
    # Accessor for the main loop
    # ----------------------------------------------------------------------- #

    def get_target(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._target.copy() if self._target is not None else None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(args=None) -> None:
    rclpy.init(args=args)

    node = FrankaMujocoNode()

    # Initialise MuJoCo env BEFORE starting the viewer loop
    xml_path = node._xml_path
    joint_names_mj = node._joint_names_mj
    step_rate = node._step_rate

    node.get_logger().info("Initialising MuJoCo FrankaEnv ...")
    env = FrankaEnv(
        xml_path=xml_path,
        joint_names=joint_names_mj,
        render=True,
    )
    env.launch_viewer()
    node.get_logger().info("MuJoCo viewer is up — waiting for commands ...")

    # Spin ROS 2 in a background thread
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    dt = 1.0 / step_rate
    import time

    try:
        while rclpy.ok() and env.viewer_is_running():
            t0 = time.monotonic()

            target = node.get_target()
            if target is not None:
                env.set_joint_position_target(target)

            env.step()

            elapsed = time.monotonic() - t0
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down ...")
        env.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
