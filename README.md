# gello_ros2_franka

A clean ROS 2 package for teleoperation of Franka robots using GELLO (a Dynamixel-based leader arm).

## Overview

This repo provides two ROS 2 nodes:

1. **`gello_driver`** — Reads joint positions from GELLO (Dynamixel servos) and publishes them as `sensor_msgs/JointState` on `/gello/joint_command`.
2. **`franka_mujoco_receiver`** — Receives the joint commands and drives a simulated Franka Emika robot in MuJoCo in real-time. Use this to validate teleoperation before running on real hardware.

```
GELLO (Dynamixel) ──► gello_driver node ──► /gello/joint_command ──► franka_mujoco_receiver node ──► MuJoCo Franka
```

## Package Layout

```
gello_ros2_franka/
├── gello_driver/                  # ROS 2 package: reads Dynamixel, publishes JointCommand
│   ├── gello_driver/
│   │   ├── dynamixel_driver.py    # Low-level Dynamixel SDK wrapper (threaded)
│   │   ├── gello_node.py          # ROS 2 node entry point
│   │   └── config.py              # PORT_CONFIG_MAP & GelloConfig dataclass
│   ├── config/
│   │   └── gello_params.yaml      # Default ROS 2 parameters
│   ├── launch/
│   │   └── gello.launch.py
│   ├── setup.py
│   └── package.xml
│
├── franka_mujoco_receiver/        # ROS 2 package: MuJoCo Franka sim receiver
│   ├── franka_mujoco_receiver/
│   │   ├── mujoco_node.py         # ROS 2 node + MuJoCo viewer
│   │   └── franka_env.py          # MuJoCo environment helper
│   ├── assets/
│   │   └── franka/                # Franka MuJoCo XML (auto-downloaded)
│   ├── launch/
│   │   └── mujoco_receiver.launch.py
│   ├── setup.py
│   └── package.xml
│
├── requirements.txt
└── README.md
```

## Prerequisites

- ROS 2 Humble (or later)
- Python ≥ 3.10
- `dynamixel_sdk` (`pip install dynamixel-sdk`)
- `mujoco` (`pip install mujoco`)
- `mujoco-mjx` is optional (GPU acceleration)

```bash
pip install -r requirements.txt
```

## Build

```bash
cd ~/ros2_ws/src
ln -s /path/to/gello_ros2_franka/gello_driver .
ln -s /path/to/gello_ros2_franka/franka_mujoco_receiver .
cd ~/ros2_ws
colcon build --packages-select gello_driver franka_mujoco_receiver
source install/setup.bash
```

Or, for development (no colcon needed):

```bash
pip install -e gello_driver/ -e franka_mujoco_receiver/
```

## Configuration

Edit `gello_driver/config/gello_params.yaml` or override via the launch file / command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `port` | `/dev/ttyUSB0` | Serial port to GELLO device |
| `baudrate` | `57600` | Dynamixel baudrate |
| `joint_ids` | `[1,2,3,4,5,6,7]` | Dynamixel servo IDs |
| `joint_offsets` | `[...]` | Per-joint offset in radians (multiples of π/2) |
| `joint_signs` | `[1,-1,1,1,1,-1,1]` | Per-joint sign correction |
| `publish_rate` | `50` | Publishing frequency (Hz) |
| `use_fake_driver` | `false` | Use fake driver when hardware not connected |

The `PORT_CONFIG_MAP` in `gello_driver/gello_driver/config.py` lets you pre-register full configurations keyed by the port's `by-id` path.

## Running

### Terminal 1 — GELLO driver

```bash
ros2 launch gello_driver gello.launch.py port:=/dev/serial/by-id/<your-device>
```

### Terminal 2 — MuJoCo Franka receiver

```bash
ros2 launch franka_mujoco_receiver mujoco_receiver.launch.py
```

A MuJoCo viewer window will open. Move the GELLO arm and the Franka in the simulator will follow.

## Topic Reference

| Topic | Type | Direction |
|-------|------|-----------|
| `/gello/joint_command` | `sensor_msgs/JointState` | gello_driver → receiver |

## License

MIT
