from lerobot.common.robot_devices.robots.configs import (
    So100RobotConfig,
)
# Import ManipulatorRobot from lerobot.common.robot_devices.robots.manipulator
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot

config = So100RobotConfig(cameras={})

robot = ManipulatorRobot(config)

robot.connect()
