import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np
import threading
import time
from typing import List, Optional

from .franka_env import FrankaEnv

class FrankaMujocoNode(Node):
    def __init__(self):
        super().__init__("franka_mujoco_node")
        
        # ------------------------------------------------------------------ #
        # Parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("joint_names_mj", [f"fr3_joint{i}" for i in range(1, 8)])
        
        self._joint_names_mj = self.get_parameter("joint_names_mj").value

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

    # ROS spinning in a separate thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Simulation setup
    env = FrankaEnv(render=True)
    
    # Real-time synchronization state
    start_time = time.time()
    sim_start_time = env._data.time
    
    node.get_logger().info("Starting ASYNC real-time simulation loop...")
    
    try:
        while rclpy.ok():
            # 1. Update target for the environment
            latest_target = node.get_target()
            if latest_target is not None:
                env.set_joint_position_target(latest_target)
            
            # 2. Synchronize simulation time with wall-clock time
            wall_time_elapsed = time.time() - start_time
            sim_time_elapsed = env._data.time - sim_start_time
            
            # If simulation is behind reality, catch up
            # We limit catch-up steps per frame to avoid "spiral of death" 
            # if the computer is way too slow.
            max_steps_per_frame = 20
            steps = 0
            while (env._data.time - sim_start_time) < wall_time_elapsed and steps < max_steps_per_frame:
                env.step()
                steps += 1
            
            # 3. Render and sync viewer sparingly (approx 60-100Hz for visual comfort)
            env.sync_viewer()
            
            # Short sleep to prevent 100% CPU usage while maintaining high responsiveness
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
