from lerobot.common.robot_devices.robots.configs import (
    So100RobotConfig,
)
# Import ManipulatorRobot from lerobot.common.robot_devices.robots.manipulator
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.utils.utils import has_method, init_logging, log_say

config = So100RobotConfig(cameras={})

robot = ManipulatorRobot(config)

robot.connect()

@dataclass
class HackMainConfig:
    model_repo_id: str = 'helper2424/pi0_shity_version'
    model_branch: str | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@draccus.wrap()
def main(cfg: HackMainConfig):
    # Initialize robot

    config = So100RobotConfig()
    robot = ManipulatorRobot(config)
    robot.connect()

    # Initialize pi0 policy
    logger.info("The robot is connected")
    log_say("THe robot is connected")
    policy = Policy.from_pretrained(
        cfg.model_repo_id,
        device=cfg.device
    )

    policy.eval()

    # Main control loop
    logger.info("Starting control loop...")
    log_say("Starting control loop...")

    try:
        while True:
            # Get robot state
            state = robot.get_state()
            
            # Get action from policy
            action = policy.get_action(state)
            
            # Execute action
            robot.execute_action(action)
            
    except KeyboardInterrupt:
        logger.info("Stopping robot...")
        robot.stop()
        logger.info("Robot stopped successfully")


if __name__ == "__main__":
    main()