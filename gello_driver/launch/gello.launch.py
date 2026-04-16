"""Launch file for gello_driver node."""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("gello_driver")
    default_params = PathJoinSubstitution([pkg_share, "config", "gello_params.yaml"])

    return LaunchDescription([
        DeclareLaunchArgument(
            "port",
            default_value="/dev/ttyUSB0",
            description="Serial port to GELLO device (prefer /dev/serial/by-id/...)",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params,
            description="Full path to the YAML parameters file",
        ),
        DeclareLaunchArgument(
            "use_fake_driver",
            default_value="false",
            description="Use fake Dynamixel driver (no hardware)",
        ),
        DeclareLaunchArgument(
            "gpio_enabled",
            default_value="true",
            description="Enable GPIO switch monitoring (Jetson Orin Nano)",
        ),
        Node(
            package="gello_driver",
            executable="gello_node",
            name="gello_node",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {
                    "port": LaunchConfiguration("port"),
                    "use_fake_driver": LaunchConfiguration("use_fake_driver"),
                    "gpio_enabled": LaunchConfiguration("gpio_enabled"),
                },
            ],
        ),
    ])
