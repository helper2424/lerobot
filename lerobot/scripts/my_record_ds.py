import time
from lerobot.scripts.control_robot import busy_wait
from lerobot.common.robot_devices.robots.manipulator import ManipulatorRobot
from lerobot.common.robot_devices.motors.dynamixel import DynamixelMotorsBus
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera


record_time_s = 30
fps = 60

states = []
actions = []

leader_port = "/dev/tty.usbmodem58FD0163901"
follower_port = "/dev/tty.usbmodem58FD0173601"

leader_arm = DynamixelMotorsBus(
    port=leader_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl330-m077"),
        "shoulder_lift": (2, "xl330-m077"),
        "elbow_flex": (3, "xl330-m077"),
        "wrist_flex": (4, "xl330-m077"),
        "wrist_roll": (5, "xl330-m077"),
        "gripper": (6, "xl330-m077"),
    },
)

follower_arm = DynamixelMotorsBus(
    port=follower_port,
    motors={
        # name: (index, model)
        "shoulder_pan": (1, "xl430-w250"),
        "shoulder_lift": (2, "xl430-w250"),
        "elbow_flex": (3, "xl330-m288"),
        "wrist_flex": (4, "xl330-m288"),
        "wrist_roll": (5, "xl330-m288"),
        "gripper": (6, "xl330-m288"),
    },
)

robot = ManipulatorRobot(
    robot_type="koch",
    leader_arms={"main": leader_arm},
    follower_arms={"main": follower_arm},
    calibration_dir=".cache/calibration/koch",
    cameras={
        "web": OpenCVCamera(0, fps=30, width=640, height=480),
        "phone": OpenCVCamera(3, fps=30, width=640, height=480),
    },
)

robot.connect()

for _ in range(record_time_s * fps):
    start_time = time.perf_counter()
    observation, action = robot.teleop_step(record_data=True)
    states.append(observation["observation.state"])
    actions.append(action["action"])
    dt_s = time.perf_counter() - start_time
    busy_wait(1 / fps - dt_s)


DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/koch_move_obj \
  policy=act_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/act_koch_test \
  hydra.job.name=act_koch_test \
  device=mps \
  wandb.enable=true


python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj \
  --tags tutorial \
  --warmup-time-s 2 \
  --episode-time-s 20 \
  --reset-time-s 10 \
  --num-episodes 50


python lerobot/scripts/visualize_dataset_html.py \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj


python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj \
  --tags tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 10 \
  -p outputs/train/act_koch_test/checkpoints/last/pretrained_model


python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/eval_koch_move_obj2 \
  --tags tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 5 \
  -p outputs/train/act_koch_test/checkpoints/last/pretrained_model


-----
# Run first episodes

python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj_test \
  --tags tutorial \
  --warmup-time-s 2 \
  --episode-time-s 20 \
  --reset-time-s 5 \
  --num-episodes 2


python lerobot/scripts/visualize_dataset_html.py \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj_static_cameras


python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj_static_cameras \
  --tags tutorial \
  --warmup-time-s 2 \
  --episode-time-s 20 \
  --reset-time-s 10 \
  --num-episodes 100


DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/koch_move_obj_static_cameras \
  policy=act_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/koch_move_obj_static_cameras \
  hydra.job.name=koch_move_obj_static_cameras \
  device=cuda \
  wandb.enable=true


huggingface-cli upload "helper2424/koch_move_obj_static_cameras" data/helper2424/koch_move_obj_static_cameras --repo-type dataset 


huggingface-cli upload ${HF_USER}/koch_move_obj_static_cameras \
  outputs/train/koch_move_obj_static_cameras/checkpoints/last/pretrained_model


python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/koch.yaml \
  --fps 30 \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj_static_cameras_eval2 \
  --tags tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 30 \
  --reset-time-s 30 \
  --num-episodes 1 \
--policy-overrides "device=mps" \
  -p ${HF_USER}/koch_move_obj_static_cameras 


python lerobot/scripts/visualize_dataset.py \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj_static_cameras_eval --episode-index 0
  

python lerobot/scripts/visualize_dataset.py \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj_static_cameras --episode-index 0

python lerobot/scripts/visualize_dataset.py \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj_static_cameras --episode-index 50

python lerobot/scripts/visualize_dataset.py \
  --root data \
  --repo-id ${HF_USER}/koch_move_obj_static_cameras_eval2 --episode-index 0


 python lerobot/scripts/control_robot.py teleoperate \
  --robot-path lerobot/configs/robot/koch.yaml \ 
  --robot-overrides '~cameras'