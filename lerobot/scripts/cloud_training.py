cd /
mkdir -p /app
cd app

git clone https://github.com/helper2424/lerobot.git
cd lerobot
conda create -y -n lerobot python=3.10
conda activate lerobot
pip install -e ".[dynamixel]"
conda install -c conda-forge ffmpeg
pip uninstall opencv-python
conda install -c conda-forge "opencv>=4.10.0"

pip install -U "huggingface_hub[cli]"

ssh-keygen -t rsa -b 4096 -C "helper2424@gmail.com"

cat ~/.ssh/id_rsa.pub
# to https://huggingface.co/settings/keys
export HUGGINGFACE_TOKEN=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

huggingface-cli download ${HF_USER}/koch_move_obj_static_cameras --repo-type dataset 

apt-get install git-lfs
git lfs install

cd data/helper2424/
git clone git@hf.co:datasets/helper2424/koch_move_obj_static_cameras

DATA_DIR=data python lerobot/scripts/train.py \
  dataset_repo_id=${HF_USER}/koch_move_obj_static_cameras \
  policy=act_koch_real \
  env=koch_real \
  hydra.run.dir=outputs/train/koch_move_obj_static_cameras \
  hydra.job.name=koch_move_obj_static_cameras \
  device=cuda \
  wandb.enable=true


git clone https://huggingface.co/helper2424/koch_move_obj_static_cameras 