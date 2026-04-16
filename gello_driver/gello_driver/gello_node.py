"""
gello_node.py — ROS 2 node: reads GELLO Dynamixel servos and publishes
joint commands on /gello/joint_command (sensor_msgs/JointState).

Also monitors up to N GPIO pins (Jetson Orin Nano) with hardware pull-up
resistors for rising-edge events and publishes std_msgs/Bool on
per-switch configurable topics.

Usage
-----
# Via launch file (recommended):
ros2 launch gello_driver gello.launch.py

# Directly (parameters from YAML or command line):
ros2 run gello_driver gello_node
"""

from __future__ import annotations

import math
import threading
from typing import Dict, List, Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from gello_driver.config import PORT_CONFIG_MAP, GelloConfig
from gello_driver.dynamixel_driver import DynamixelDriver, FakeDynamixelDriver
from gello_driver.gpio_switch_handler import GpioSwitchHandler


# Default joint names for a 7-DOF Franka FR3 / Panda
FRANKA_JOINT_NAMES = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]


class GelloNode(Node):
    """
    ROS 2 node that polls a GELLO device and publishes JointState commands.

    Parameters (declared as ROS 2 params, can be set in YAML or CLI)
    ----------
    port              : str   — serial device path
    baudrate          : int   — Dynamixel baudrate
    joint_ids         : list  — servo IDs (arm joints only)
    joint_offsets     : list  — per-joint offset (rad)
    joint_signs       : list  — per-joint sign (+1 or -1)
    joint_names       : list  — joint names for the published JointState
    gripper_id        : int   — gripper servo ID (-1 = no gripper)
    gripper_open      : float — gripper open position (deg)
    gripper_closed    : float — gripper closed position (deg)
    alpha             : float — smoothing factor (0–1; 1 = no smoothing)
    publish_rate      : float — Hz
    use_fake_driver   : bool  — use fake driver (no hardware required)
    auto_detect_port  : bool  — look up port in PORT_CONFIG_MAP

    GPIO switch parameters
    ----------------------
    gpio_enabled      : bool  — enable GPIO switch monitoring (default False)
    gpio_pin_mode     : str   — 'BOARD' (physical) or 'BCM'
    gpio_bouncetime_ms: int   — debounce time in milliseconds
    gpio_pins         : list  — BOARD pin numbers for switches [sw0, sw1, sw2]
    gpio_topic_names  : list  — ROS 2 topic names for each switch event
    """

    def __init__(self) -> None:
        super().__init__("gello_node")

        # ------------------------------------------------------------------ #
        # Declare parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 57600)
        self.declare_parameter("joint_ids", [1, 2, 3, 4, 5, 6, 7])
        self.declare_parameter("joint_offsets", [
            3 * math.pi / 2,
            2 * math.pi / 2,
            1 * math.pi / 2,
            4 * math.pi / 2,
            -2 * math.pi / 2 + 2 * math.pi,
            3 * math.pi / 2,
            4 * math.pi / 2,
        ])
        self.declare_parameter("joint_signs", [1, -1, 1, 1, 1, -1, 1])
        self.declare_parameter("joint_names", FRANKA_JOINT_NAMES)
        self.declare_parameter("gripper_id", -1)          # -1 = no gripper
        self.declare_parameter("gripper_open", 195.0)     # degrees
        self.declare_parameter("gripper_closed", 152.0)   # degrees
        self.declare_parameter("alpha", 0.5)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("use_fake_driver", False)
        self.declare_parameter("auto_detect_port", True)

        # GPIO switches
        self.declare_parameter("gpio_enabled", False)
        self.declare_parameter("gpio_pin_mode", "BOARD")
        self.declare_parameter("gpio_bouncetime_ms", 50)
        # Default: three pins commonly available on Jetson Orin Nano 40-pin header
        self.declare_parameter("gpio_pins", [7, 11, 13])
        self.declare_parameter("gpio_topic_names", [
            "/gello/switch/0",
            "/gello/switch/1",
            "/gello/switch/2",
        ])

        # ------------------------------------------------------------------ #
        # Resolve configuration
        # ------------------------------------------------------------------ #
        self._cfg = self._resolve_config()

        # ------------------------------------------------------------------ #
        # Initialise driver
        # ------------------------------------------------------------------ #
        use_fake = self.get_parameter("use_fake_driver").get_parameter_value().bool_value
        all_ids = self._cfg.all_ids

        if use_fake:
            self._driver = FakeDynamixelDriver(all_ids)
            self.get_logger().warn("Using FakeDynamixelDriver — no hardware required.")
        else:
            self._driver = DynamixelDriver(
                ids=all_ids,
                port=self._cfg.port if hasattr(self._cfg, "port") else
                     self.get_parameter("port").get_parameter_value().string_value,
                baudrate=self._cfg.baudrate,
                use_fake_fallback=True,
            )
            if self._driver.is_fake:
                self.get_logger().warn(
                    "Hardware not found — running with FakeDynamixelDriver."
                )
            else:
                self.get_logger().info("Dynamixel driver initialised on real hardware.")

        self._driver.set_torque_mode(False)  # GELLO = leader, torque off

        # ------------------------------------------------------------------ #
        # Internal state
        # ------------------------------------------------------------------ #
        self._joint_offsets = np.array(self._cfg.joint_offsets)
        self._joint_signs = np.array(self._cfg.joint_signs)
        self._alpha = self._cfg.alpha
        self._last_pos: Optional[np.ndarray] = None
        self._joint_names: List[str] = (
            self.get_parameter("joint_names").get_parameter_value().string_array_value
        )

        # ------------------------------------------------------------------ #
        # JointCommand publisher
        # ------------------------------------------------------------------ #
        self._pub = self.create_publisher(JointState, "/gello/joint_command", 10)

        rate = self.get_parameter("publish_rate").get_parameter_value().double_value
        self._timer = self.create_timer(1.0 / rate, self._publish_cb)

        self.get_logger().info(
            f"GelloNode started — publishing at {rate:.0f} Hz on /gello/joint_command"
        )

        # ------------------------------------------------------------------ #
        # GPIO switch publishers + handler
        # ------------------------------------------------------------------ #
        self._switch_pubs: Dict[str, rclpy.publisher.Publisher] = {}
        self._gpio_handler: Optional[GpioSwitchHandler] = None
        self._gpio_pub_lock = threading.Lock()

        if self.get_parameter("gpio_enabled").get_parameter_value().bool_value:
            self._setup_gpio()
        else:
            self.get_logger().info("GPIO switches disabled (gpio_enabled=false).")

    # ----------------------------------------------------------------------- #
    # Config resolution
    # ----------------------------------------------------------------------- #

    def _resolve_config(self) -> GelloConfig:
        """
        If auto_detect_port is True and the port is in PORT_CONFIG_MAP,
        use the registered GelloConfig (ignoring individual params).
        Otherwise, build a GelloConfig from individual ROS 2 parameters.
        """
        port = self.get_parameter("port").get_parameter_value().string_value
        auto = self.get_parameter("auto_detect_port").get_parameter_value().bool_value

        if auto and port in PORT_CONFIG_MAP:
            cfg = PORT_CONFIG_MAP[port]
            self.get_logger().info(
                f"Auto-detected config for port {port}"
            )
            # Attach the port string so the driver can use it
            cfg.__dict__["port"] = port
            return cfg

        # Build from params
        joint_ids = list(
            self.get_parameter("joint_ids").get_parameter_value().integer_array_value
        )
        joint_offsets = list(
            self.get_parameter("joint_offsets").get_parameter_value().double_array_value
        )
        joint_signs = [
            int(s)
            for s in self.get_parameter("joint_signs").get_parameter_value().integer_array_value
        ]
        gripper_id_val = self.get_parameter("gripper_id").get_parameter_value().integer_value
        gripper_id = gripper_id_val if gripper_id_val > 0 else None

        cfg = GelloConfig(
            joint_ids=joint_ids,
            joint_offsets=joint_offsets,
            joint_signs=joint_signs,
            gripper_id=gripper_id,
            gripper_open_pos_deg=self.get_parameter("gripper_open").get_parameter_value().double_value,
            gripper_closed_pos_deg=self.get_parameter("gripper_closed").get_parameter_value().double_value,
            baudrate=self.get_parameter("baudrate").get_parameter_value().integer_value,
            alpha=self.get_parameter("alpha").get_parameter_value().double_value,
        )
        cfg.__dict__["port"] = port
        return cfg

    # ----------------------------------------------------------------------- #
    # Timer callback
    # ----------------------------------------------------------------------- #

    def _publish_cb(self) -> None:
        try:
            raw = self._driver.get_joints()          # all servo IDs
        except Exception as exc:
            self.get_logger().warn(f"get_joints failed: {exc}")
            return

        # --- Apply calibration to arm joints only --------------------------
        arm_raw = raw[: self._cfg.n_joints]
        arm_pos = (arm_raw - self._joint_offsets) * self._joint_signs

        # Exponential smoothing
        if self._last_pos is None:
            self._last_pos = arm_pos.copy()
        else:
            arm_pos = self._last_pos * (1.0 - self._alpha) + arm_pos * self._alpha
            self._last_pos = arm_pos.copy()

        # Optionally compute normalised gripper [0, 1]
        gripper_norm: Optional[float] = None
        if self._cfg.gripper_id is not None and len(raw) > self._cfg.n_joints:
            gr = self._cfg.gripper_range_rad
            g_raw = raw[self._cfg.n_joints]
            if gr is not None and abs(gr[1] - gr[0]) > 1e-6:
                gripper_norm = float(np.clip(
                    (g_raw - gr[0]) / (gr[1] - gr[0]), 0.0, 1.0
                ))

        # --- Build and publish message -------------------------------------
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.name = list(self._joint_names[: self._cfg.n_joints])
        msg.position = arm_pos.tolist()

        if gripper_norm is not None:
            msg.name.append("finger_joint")
            msg.position.append(gripper_norm)

        self._pub.publish(msg)

    # ----------------------------------------------------------------------- #
    # GPIO setup
    # ----------------------------------------------------------------------- #

    def _setup_gpio(self) -> None:
        gpio_pins = list(
            self.get_parameter("gpio_pins").get_parameter_value().integer_array_value
        )
        gpio_topics = list(
            self.get_parameter("gpio_topic_names").get_parameter_value().string_array_value
        )

        if len(gpio_pins) != len(gpio_topics):
            self.get_logger().error(
                f"gpio_pins ({len(gpio_pins)}) and gpio_topic_names "
                f"({len(gpio_topics)}) must have the same length. "
                "GPIO switches will NOT be set up."
            )
            return

        # Create one Bool publisher per switch topic
        for topic in gpio_topics:
            self._switch_pubs[topic] = self.create_publisher(Bool, topic, 10)
            self.get_logger().info(f"GPIO switch publisher: {topic}")

        pin_configs = list(zip(gpio_pins, gpio_topics))
        pin_mode = (
            self.get_parameter("gpio_pin_mode").get_parameter_value().string_value
        )
        bouncetime = (
            self.get_parameter("gpio_bouncetime_ms").get_parameter_value().integer_value
        )

        self._gpio_handler = GpioSwitchHandler(
            pin_configs=pin_configs,
            callback=self._gpio_edge_cb,
            pin_mode=pin_mode,
            bouncetime_ms=bouncetime,
            logger=self.get_logger(),
        )

    def _gpio_edge_cb(self, topic_name: str) -> None:
        """
        Called from a GPIO interrupt thread when a rising edge fires.
        Publishes Bool(True) on the corresponding topic.

        Note: GPIO callbacks run in a C-level thread that is NOT the ROS 2
        executor thread.  Publishing from here is safe in rclpy because
        the publisher is thread-safe, but we guard with a lock to be tidy.
        """
        with self._gpio_pub_lock:
            pub = self._switch_pubs.get(topic_name)
            if pub is None:
                return
            msg = Bool()
            msg.data = True
            pub.publish(msg)
        self.get_logger().info(f"GPIO rising edge → published on {topic_name}")

    # ----------------------------------------------------------------------- #
    # Cleanup
    # ----------------------------------------------------------------------- #

    def destroy_node(self) -> None:
        if self._gpio_handler is not None:
            self._gpio_handler.cleanup()
        self._driver.close()
        super().destroy_node()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = GelloNode()
    except Exception as exc:
        print(f"[gello_node] Failed to start: {exc}")
        rclpy.try_shutdown()
        return

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
