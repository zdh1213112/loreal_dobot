from setuptools import find_packages, setup


package_name = "dobot_nova5_driver"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/nova5_driver.launch.py"]),
        (
            f"share/{package_name}/dobot_nova5_driver/TCP_IP_Python_V4/files",
            [
                "dobot_nova5_driver/TCP_IP_Python_V4/files/alarmController.json",
                "dobot_nova5_driver/TCP_IP_Python_V4/files/alarmServo.json",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zdh",
    maintainer_email="zhangdaohu@xenserobotics.com",
    description="ROS 2 Python driver package for Dobot Nova5 TCP/IP control.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "nova5_driver_node = dobot_nova5_driver.nova5_driver_node:main",
        ],
    },
)
