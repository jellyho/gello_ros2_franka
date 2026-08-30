"""gello_driver ROS 2 package setup."""

from setuptools import find_packages, setup

package_name = "gello_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/gello_params.yaml"]),
        (f"share/{package_name}/launch", ["launch/gello.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jellyho",
    maintainer_email="jellyho@todo.todo",
    description="ROS 2 driver node for GELLO Dynamixel teleoperation device",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "gello_node = gello_driver.gello_node:main",
            "calibrate_offsets = gello_driver.calibrate_offsets:main",
        ],
    },
)
