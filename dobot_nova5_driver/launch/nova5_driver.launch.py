from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="dobot_nova5_driver",
                executable="nova5_driver_node",
                name="nova5_driver_node",
                output="screen",
            )
        ]
    )
