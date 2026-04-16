"""Launch file for franka_mujoco_receiver node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "xml_path",
            default_value="",
            description=(
                "Absolute path to fr3.xml.  Leave empty to auto-download "
                "from mujoco_menagerie."
            ),
        ),
        DeclareLaunchArgument(
            "step_rate",
            default_value="50.0",
            description="Physics / viewer update rate (Hz)",
        ),
        Node(
            package="franka_mujoco_receiver",
            executable="mujoco_node",
            name="franka_mujoco_receiver",
            output="screen",
            parameters=[
                {
                    "xml_path": LaunchConfiguration("xml_path"),
                    "step_rate": LaunchConfiguration("step_rate"),
                    "joint_names_mj": [
                        "joint1",
                        "joint2",
                        "joint3",
                        "joint4",
                        "joint5",
                        "joint6",
                        "joint7",
                    ],
                }
            ],
        ),
    ])
