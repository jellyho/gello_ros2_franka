import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import threading
from typing import List, Optional

from .franka_env import FrankaEnv

class FrankaMujocoNode(Node):
    def __init__(self):
        super().__init__("franka_mujoco_node")
        
        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("joint_names_mj", [f"fr3_joint{i}" for i in range(1, 8)])
        self.declare_parameter("step_rate", 500.0) # Higher Hz for better responsiveness
        
        self._joint_names_mj = self.get_parameter("joint_names_mj").value
        self._step_rate = self.get_parameter("step_rate").value

        # ------------------------------------------------------------------ #
        # State
        # ------------------------------------------------------------------ #
        self._lock = threading.Lock()
        self._target: Optional[np.ndarray] = None
        self._n_joints = len(self._joint_names_mj)
        self._msg_count = 0

        # ------------------------------------------------------------------ #
        # ROS 2 Interface
        # ------------------------------------------------------------------ #
        self._sub = self.create_subscription(
            JointState,
            "/gello/joint_command",
            self._joint_command_cb,
            10
        )
        
        self.get_logger().info(f"FrankaMujocoNode listening on /gello/joint_command")

    def _joint_command_cb(self, msg: JointState) -> None:
        # Debug: Print something as soon as ANY message hits the callback
        if self._msg_count == 0:
            self.get_logger().info("!!! CALLBACK TRIGGERED !!! First message arrived.")
            
        if len(msg.name) == 0 or len(msg.position) == 0:
            return

        name_to_pos = {name: pos for name, pos in zip(msg.name, msg.position)}
        positions = np.zeros(self._n_joints)
        
        for i, mj_name in enumerate(self._joint_names_mj):
            ros_name = f"fr3_{mj_name}" if not mj_name.startswith("fr3_") else mj_name
            if mj_name in name_to_pos:
                positions[i] = name_to_pos[mj_name]
            elif ros_name in name_to_pos:
                positions[i] = name_to_pos[ros_name]

        with self._lock:
            self._target = positions
            self._msg_count += 1
            if self._msg_count % 100 == 0:
                self.get_logger().info(f"Received {self._msg_count} commands. Target: {positions.round(3)}")

    def get_target(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._target.copy() if self._target is not None else None

def main():
    rclpy.init()
    node = FrankaMujocoNode()

    # Create a separate thread for ROS spinning to ensure messages are never missed
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Local simulation loop
    env = FrankaEnv(render=True)
    rate = node.create_rate(node._step_rate)
    
    try:
        node.get_logger().info("Starting simulation loop...")
        while rclpy.ok():
            target = node.get_target()
            if target is not None:
                env.set_joint_position_target(target)
            
            # Pass the delta time (1/rate) to the step function
            env.step(1.0 / node._step_rate)
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
