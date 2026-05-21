"""
gello_node.py — ROS 2 node: reads GELLO Dynamixel servos and publishes
joint commands on /gello/joint_command (sensor_msgs/JointState).

Also monitors up to N GPIO pins (Jetson Orin Nano) with hardware pull-up
resistors and publishes std_msgs/Bool true pulses on per-switch configurable
topics when buttons are pressed.

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

from gello_driver.dynamixel_driver import (
    DynamixelDriver,
    FakeDynamixelDriver,
    CURRENT_BASED_POSITION_CTRL_MODE,
)
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
    gpio_pin_mode     : str   — 'BOARD' for physical header pins
    gpio_pins         : list  — BOARD pin numbers for switches
    gpio_bouncetime_ms: int   — debounce time in milliseconds
    gpio_topic_names  : list  — ROS 2 topic names for each button press
    """

    def __init__(self) -> None:
        super().__init__("gello_node")

        # ------------------------------------------------------------------ #
        # Declare parameters
        # ------------------------------------------------------------------ #
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 57600)
        self.declare_parameter("joint_ids", [1, 2, 3, 4, 5, 6, 7])
        self.declare_parameter("joint_offsets", [270.0, 180.0, 90.0, 360.0, 180.0, 270.0, 360.0])  # in degrees
        self.declare_parameter("joint_signs", [1, -1, 1, 1, 1, -1, 1])
        self.declare_parameter("joint_names", FRANKA_JOINT_NAMES)
        self.declare_parameter("gripper_id", -1)          # -1 = no gripper
        self.declare_parameter("gripper_open", 195.0)     # degrees
        self.declare_parameter("gripper_closed", 152.0)   # degrees
        self.declare_parameter("gripper_hold", 175.0)     # degrees — hold / rest position
        self.declare_parameter("alpha", 0.5)
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("use_fake_driver", False)

        # GPIO switches
        self.declare_parameter("gpio_enabled", False)
        self.declare_parameter("gpio_pin_mode", "BOARD")
        self.declare_parameter("gpio_bouncetime_ms", 250)
        self.declare_parameter("gpio_confirm_delay_ms", 20)
        self.declare_parameter("gpio_pins", [7, 11, 13])
        self.declare_parameter("gpio_topic_names", [
            "/gello/switch/0",
            "/gello/switch/1",
            "/gello/switch/2",
        ])

        # ------------------------------------------------------------------ #
        # Load parameters into members
        # ------------------------------------------------------------------ #
        port = self.get_parameter("port").get_parameter_value().string_value
        baudrate = self.get_parameter("baudrate").get_parameter_value().integer_value
        self._joint_ids = list(
            self.get_parameter("joint_ids").get_parameter_value().integer_array_value
        )
        # Load joint offsets in degrees and convert to radians
        joint_offsets_deg = self.get_parameter("joint_offsets").get_parameter_value().double_array_value
        self._joint_offsets = np.radians(joint_offsets_deg)
        self._joint_signs = np.array([
            int(s)
            for s in self.get_parameter("joint_signs").get_parameter_value().integer_array_value
        ])
        self._joint_names = self.get_parameter("joint_names").get_parameter_value().string_array_value

        gripper_id_val = self.get_parameter("gripper_id").get_parameter_value().integer_value
        self._gripper_id = gripper_id_val if gripper_id_val >= 0 else None
        self._gripper_open_deg = self.get_parameter("gripper_open").get_parameter_value().double_value
        self._gripper_closed_deg = self.get_parameter("gripper_closed").get_parameter_value().double_value
        self._gripper_hold_deg = self.get_parameter("gripper_hold").get_parameter_value().double_value

        self._alpha = self.get_parameter("alpha").get_parameter_value().double_value
        publish_rate = self.get_parameter("publish_rate").get_parameter_value().double_value

        # Helper list for driver initialization
        all_ids = list(self._joint_ids)
        if self._gripper_id is not None:
            all_ids.append(self._gripper_id)

        # ------------------------------------------------------------------ #
        # Initialise driver
        # ------------------------------------------------------------------ #
        use_fake = self.get_parameter("use_fake_driver").get_parameter_value().bool_value

        if use_fake:
            self._driver = FakeDynamixelDriver(all_ids)
            self.get_logger().warn("Using FakeDynamixelDriver")
        else:
            self._driver = DynamixelDriver(
                ids=all_ids,
                port=port,
                baudrate=baudrate,
                use_fake_fallback=True,
            )
            if self._driver.is_fake:
                self.get_logger().warn("Hardware not found — running with FakeDynamixelDriver.")
            else:
                self.get_logger().info(f"Dynamixel driver initialised on {port} at {baudrate} baud.")

        # ------------------------------------------------------------------ #
        # Startup hardware sequence
        # ──────────────────────────────────────────────────────────────────
        # 1. Ensure ALL motors torque-off (required to change operating mode)
        # 2. Turn on LEDs for all motors  (visual feedback)
        # 3. Arm joints: stay torque-off  (GELLO = passive leader)
        # 4. Gripper:   position-control mode → torque on → hold position
        # ------------------------------------------------------------------ #
        self._driver.set_torque_mode(False)          # step 1: all off
        self._driver.set_led(True)                   # step 2: LEDs on
        self.get_logger().info("Motor LEDs activated.")

        self._init_active_joints()

        # ------------------------------------------------------------------ #
        # Internal control state
        # ------------------------------------------------------------------ #
        self._last_pos: Optional[np.ndarray] = None

        # ------------------------------------------------------------------ #
        # JointCommand publisher
        # ------------------------------------------------------------------ #
        self._pub = self.create_publisher(JointState, "/gello/joint_command", 10)

        self._timer = self.create_timer(1.0 / publish_rate, self._publish_cb)

        self.get_logger().info(
            f"GelloNode started — publishing at {publish_rate:.1f} Hz on /gello/joint_command"
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
    # Hardware initialization for active joints (Gripper + Shoulder)
    # ----------------------------------------------------------------------- #

    def _init_active_joints(self) -> None:
        """
        Configure active joints for Current-based Position Control Mode (Mode 5).
        """
        # --- 1. Base and Shoulder Joint (Second from base, index 1) ---
        if len(self._joint_ids) > 1:
            sid = self._joint_ids[0]
            hold_rad = self._joint_offsets[0]
            try:
                self._driver.set_operating_mode_ids([sid], CURRENT_BASED_POSITION_CTRL_MODE)
                self._driver.set_torque_mode_ids([sid], True)
                self._driver.set_goal_position_single(sid, hold_rad)
                self.get_logger().info(
                    f"Arm Joint ID {sid} (Base): Mode 5 ON, holding at {math.degrees(hold_rad)}°."
                )
            except Exception as exc:
                self.get_logger().error(f"Failed to init shoulder joint ID {sid}: {exc}")

            sid = self._joint_ids[1]
            hold_rad = self._joint_offsets[1]
            try:
                self._driver.set_operating_mode_ids([sid], CURRENT_BASED_POSITION_CTRL_MODE)
                self._driver.set_torque_mode_ids([sid], True)
                self._driver.set_goal_position_single(sid, hold_rad)
                self.get_logger().info(
                    f"Arm Joint ID {sid} (Shoulder): Mode 5 ON, holding at {math.degrees(hold_rad)}°."
                )
            except Exception as exc:
                self.get_logger().error(f"Failed to init shoulder joint ID {sid}: {exc}")

        # --- 2. Gripper ---
        if self._gripper_id is not None:
            gid = self._gripper_id
            hold_rad = math.radians(self._gripper_hold_deg)
            try:
                self._driver.set_operating_mode_ids([gid], CURRENT_BASED_POSITION_CTRL_MODE)
                self._driver.set_torque_mode_ids([gid], True)
                self._driver.set_goal_position_single(gid, hold_rad)
                self.get_logger().info(
                    f"Gripper ID {gid}: Mode 5 ON, holding at {self._gripper_hold_deg:.1f}°."
                )
            except Exception as exc:
                self.get_logger().error(f"Failed to init gripper ID {gid}: {exc}")
        else:
            self.get_logger().info("No gripper configured.")

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
        n_arm = len(self._joint_ids)
        arm_raw = raw[:n_arm]
        arm_pos = (arm_raw - self._joint_offsets) * self._joint_signs

        # Exponential smoothing
        if self._last_pos is None:
            self._last_pos = arm_pos.copy()
        else:
            arm_pos = self._last_pos * (1.0 - self._alpha) + arm_pos * self._alpha
            self._last_pos = arm_pos.copy()

        # Optionally compute normalised gripper [0, 1]
        gripper_norm: Optional[float] = None
        if self._gripper_id is not None and len(raw) > n_arm:
            g_raw = raw[n_arm]
            g_open = math.radians(self._gripper_open_deg)
            g_close = math.radians(self._gripper_closed_deg)
            if abs(g_close - g_open) > 1e-6:
                gripper_norm = float(np.clip(
                    (g_raw - g_open) / (g_close - g_open), 0.0, 1.0
                ))

        # --- Build and publish message -------------------------------------
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"
        msg.name = list(self._joint_names[:n_arm])
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
        confirm_delay = (
            self.get_parameter("gpio_confirm_delay_ms").get_parameter_value().integer_value
        )
        self._gpio_handler = GpioSwitchHandler(
            pin_configs=pin_configs,
            callback=self._gpio_edge_cb,
            pin_mode=pin_mode,
            bouncetime_ms=bouncetime,
            confirm_delay_ms=confirm_delay,
            logger=self.get_logger(),
        )

    def _gpio_edge_cb(self, topic_name: str, pressed: bool) -> None:
        """
        Called from a GPIO interrupt thread when a button is pressed.
        Publishes Bool(True) on the corresponding topic.

        Note: GPIO callbacks run in a C-level thread that is NOT the ROS 2
        executor thread.  Publishing from here is safe in rclpy because
        the publisher is thread-safe, but we guard with a lock to be tidy.
        """
        if not pressed:
            return
        with self._gpio_pub_lock:
            pub = self._switch_pubs.get(topic_name)
            if pub is None:
                return
            msg = Bool()
            msg.data = True
            pub.publish(msg)
        self.get_logger().info(f"GPIO button pressed -> published true on {topic_name}")

    # ----------------------------------------------------------------------- #
    # Cleanup
    # ----------------------------------------------------------------------- #

    def destroy_node(self) -> None:
        # Disable torque for active joints before closing
        if self._gripper_id is not None:
            try:
                self._driver.set_torque_mode_ids([self._gripper_id], False)
            except Exception:
                pass
        if len(self._joint_ids) > 1:
            try:
                self._driver.set_torque_mode_ids([self._joint_ids[1]], False)
            except Exception:
                pass

        # Turn off all LEDs
        try:
            self._driver.set_led(False)
        except Exception:
            pass
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
