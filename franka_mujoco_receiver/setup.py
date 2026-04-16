"""franka_mujoco_receiver ROS 2 package setup."""

from setuptools import find_packages, setup

package_name = "franka_mujoco_receiver"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch",
         ["launch/mujoco_receiver.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jellyho",
    maintainer_email="jellyho@todo.todo",
    description="MuJoCo Franka simulation receiver for GELLO joint commands",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mujoco_node = franka_mujoco_receiver.mujoco_node:main",
        ],
    },
)
