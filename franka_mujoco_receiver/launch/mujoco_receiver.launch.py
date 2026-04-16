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
            default_value="500.0",
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
                        "fr3_joint1",
                        "fr3_joint2",
                        "fr3_joint3",
                        "fr3_joint4",
                        "fr3_joint5",
                        "fr3_joint6",
                        "fr3_joint7",
                    ],
                }
            ],
        ),
    ])
