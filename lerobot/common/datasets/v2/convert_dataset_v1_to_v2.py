"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 1.6 to
2.0. You will be required to provide the 'tasks', which is a short but accurate description in plain English
for each of the task performed in the dataset. This will allow to easily train models with task-conditionning.

We support 3 different scenarios for these tasks (see instructions below):
    1. Single task dataset: all episodes of your dataset have the same single task.
    2. Single task episodes: the episodes of your dataset each contain a single task but they can differ from
      one episode to the next.
    3. Multi task episodes: episodes of your dataset may each contain several different tasks.


Can you can also provide a robot config .yaml file (not mandatory) to this script via the option
'--robot-config' so that it writes information about the robot (robot type, motors names) this dataset was
recorded with. For now, only Aloha/Koch type robots are supported with this option.


# 1. Single task dataset
If your dataset contains a single task, you can simply provide it directly via the CLI with the
'--single-task' option.

Examples:

```bash
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \
    --repo-id lerobot/aloha_sim_insertion_human_image \
    --single-task "Insert the peg into the socket." \
    --robot-config lerobot/configs/robot/aloha.yaml \
    --local-dir data
```

```bash
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \
    --repo-id aliberts/koch_tutorial \
    --single-task "Pick the Lego block and drop it in the box on the right." \
    --robot-config lerobot/configs/robot/koch.yaml \
    --local-dir data
```


# 2. Single task episodes
If your dataset is a multi-task dataset, you have two options to provide the tasks to this script:

- If your dataset already contains a language instruction column in its parquet file, you can simply provide
  this column's name with the '--tasks-col' arg.

    Example:

    ```bash
    python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \
        --repo-id lerobot/stanford_kuka_multimodal_dataset \
        --tasks-col "language_instruction" \
        --local-dir data
    ```

- If your dataset doesn't contain a language instruction, you should provide the path to a .json file with the
  '--tasks-path' arg. This file should have the following structure where keys correspond to each
  episode_index in the dataset, and values are the language instruction for that episode.

    Example:

    ```json
    {
        "0": "Do something",
        "1": "Do something else",
        "2": "Do something",
        "3": "Go there",
        ...
    }
    ```

# 3. Multi task episodes
If you have multiple tasks per episodes, your dataset should contain a language instruction column in its
parquet file, and you must provide this column's name with the '--tasks-col' arg.

Example:

```bash
python lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py \
    --repo-id lerobot/stanford_kuka_multimodal_dataset \
    --tasks-col "language_instruction" \
    --local-dir data
```
"""

import argparse
import contextlib
import json
import math
import subprocess
from pathlib import Path

import datasets
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from safetensors.torch import load_file

from lerobot.common.datasets.utils import create_branch, flatten_dict, get_hub_safe_version, unflatten_dict
from lerobot.common.utils.utils import init_hydra_config
from lerobot.scripts.push_dataset_to_hub import push_dataset_card_to_hub

V16 = "v1.6"
V20 = "v2.0"

PARQUET_PATH = "data/train-{episode_index:05d}-of-{total_episodes:05d}.parquet"
VIDEO_PATH = "videos/{video_key}_episode_{episode_index:06d}.mp4"


def parse_robot_config(config_path: Path, config_overrides: list[str] | None = None) -> tuple[str, dict]:
    robot_cfg = init_hydra_config(config_path, config_overrides)
    if robot_cfg["robot_type"] in ["aloha", "koch"]:
        state_names = [
            f"{arm}_{motor}" if len(robot_cfg["follower_arms"]) > 1 else motor
            for arm in robot_cfg["follower_arms"]
            for motor in robot_cfg["follower_arms"][arm]["motors"]
        ]
        action_names = [
            # f"{arm}_{motor}" for arm in ["left", "right"] for motor in robot_cfg["leader_arms"][arm]["motors"]
            f"{arm}_{motor}" if len(robot_cfg["leader_arms"]) > 1 else motor
            for arm in robot_cfg["leader_arms"]
            for motor in robot_cfg["leader_arms"][arm]["motors"]
        ]
    # elif robot_cfg["robot_type"] == "stretch3": TODO
    else:
        raise NotImplementedError(
            "Please provide robot_config={'robot_type': ..., 'names': ...} directly to convert_dataset()."
        )

    return {
        "robot_type": robot_cfg["robot_type"],
        "names": {
            "observation.state": state_names,
            "action": action_names,
        },
    }


def load_json(fpath: Path) -> dict:
    with open(fpath) as f:
        return json.load(f)


def write_json(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4)


def convert_stats_to_json(input_dir: Path, output_dir: Path) -> None:
    safetensor_path = input_dir / "stats.safetensors"
    stats = load_file(safetensor_path)
    serialized_stats = {key: value.tolist() for key, value in stats.items()}
    serialized_stats = unflatten_dict(serialized_stats)

    json_path = output_dir / "stats.json"
    json_path.parent.mkdir(exist_ok=True, parents=True)
    with open(json_path, "w") as f:
        json.dump(serialized_stats, f, indent=4)

    # Sanity check
    with open(json_path) as f:
        stats_json = json.load(f)

    stats_json = flatten_dict(stats_json)
    stats_json = {key: torch.tensor(value) for key, value in stats_json.items()}
    for key in stats:
        torch.testing.assert_close(stats_json[key], stats[key])


def get_keys(dataset: Dataset) -> dict[str, list]:
    sequence_keys, image_keys, video_keys = [], [], []
    for key, ft in dataset.features.items():
        if isinstance(ft, datasets.Sequence):
            sequence_keys.append(key)
        elif isinstance(ft, datasets.Image):
            image_keys.append(key)
        elif ft._type == "VideoFrame":
            video_keys.append(key)

    return {
        "sequence": sequence_keys,
        "image": image_keys,
        "video": video_keys,
    }


def add_task_index_by_episodes(dataset: Dataset, tasks_by_episodes: dict) -> tuple[Dataset, list[str]]:
    df = dataset.to_pandas()
    tasks = list(set(tasks_by_episodes.values()))
    tasks_to_task_index = {task: task_idx for task_idx, task in enumerate(tasks)}
    episodes_to_task_index = {ep_idx: tasks_to_task_index[task] for ep_idx, task in tasks_by_episodes.items()}
    df["task_index"] = df["episode_index"].map(episodes_to_task_index).astype(int)

    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = Dataset.from_pandas(df, features=features, split="train")
    return dataset, tasks


def add_task_index_from_tasks_col(
    dataset: Dataset, tasks_col: str
) -> tuple[Dataset, dict[str, list[str]], list[str]]:
    df = dataset.to_pandas()

    # HACK: This is to clean some of the instructions in our version of Open X datasets
    prefix_to_clean = "tf.Tensor(b'"
    suffix_to_clean = "', shape=(), dtype=string)"
    df[tasks_col] = df[tasks_col].str.removeprefix(prefix_to_clean).str.removesuffix(suffix_to_clean)

    # Create task_index col
    tasks_by_episode = df.groupby("episode_index")[tasks_col].unique().apply(lambda x: x.tolist()).to_dict()
    tasks = df[tasks_col].unique().tolist()
    tasks_to_task_index = {task: idx for idx, task in enumerate(tasks)}
    df["task_index"] = df[tasks_col].map(tasks_to_task_index).astype(int)

    # Build the dataset back from df
    features = dataset.features
    features["task_index"] = datasets.Value(dtype="int64")
    dataset = Dataset.from_pandas(df, features=features, split="train")
    dataset = dataset.remove_columns(tasks_col)

    return dataset, tasks, tasks_by_episode


def split_parquet_by_episodes(
    dataset: Dataset, keys: dict[str, list], total_episodes: int, episode_indices: list, output_dir: Path
) -> list:
    (output_dir / "data").mkdir(exist_ok=True, parents=True)
    table = dataset.remove_columns(keys["video"])._data.table
    episode_lengths = []
    for episode_index in sorted(episode_indices):
        # Write each episode_index to a new parquet file
        filtered_table = table.filter(pc.equal(table["episode_index"], episode_index))
        episode_lengths.insert(episode_index, len(filtered_table))
        output_file = output_dir / PARQUET_PATH.format(
            episode_index=episode_index, total_episodes=total_episodes
        )
        pq.write_table(filtered_table, output_file)

    return episode_lengths


def _get_audio_info(video_path: Path | str) -> dict:
    ffprobe_audio_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels,codec_name,bit_rate,sample_rate,bit_depth,channel_layout,duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    audio_stream_info = info["streams"][0] if info.get("streams") else None
    if audio_stream_info is None:
        return {"has_audio": False}

    # Return the information, defaulting to None if no audio stream is present
    return {
        "has_audio": True,
        "audio.channels": audio_stream_info.get("channels", None),
        "audio.codec": audio_stream_info.get("codec_name", None),
        "audio.bit_rate": int(audio_stream_info["bit_rate"]) if audio_stream_info.get("bit_rate") else None,
        "audio.sample_rate": int(audio_stream_info["sample_rate"])
        if audio_stream_info.get("sample_rate")
        else None,
        "audio.bit_depth": audio_stream_info.get("bit_depth", None),
        "audio.channel_layout": audio_stream_info.get("channel_layout", None),
    }


def _get_video_info(video_path: Path | str) -> dict:
    ffprobe_video_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,width,height,codec_name,nb_frames,duration,pix_fmt",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(ffprobe_video_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Error running ffprobe: {result.stderr}")

    info = json.loads(result.stdout)
    video_stream_info = info["streams"][0]

    # Calculate fps from r_frame_rate
    r_frame_rate = video_stream_info["r_frame_rate"]
    num, denom = map(int, r_frame_rate.split("/"))
    fps = num / denom

    pixel_channels = get_video_pixel_channels(video_stream_info["pix_fmt"])

    video_info = {
        "video.fps": fps,
        "video.width": video_stream_info["width"],
        "video.height": video_stream_info["height"],
        "video.channels": pixel_channels,
        "video.codec": video_stream_info["codec_name"],
        "video.pix_fmt": video_stream_info["pix_fmt"],
        "video.is_depth_map": False,
        **_get_audio_info(video_path),
    }

    return video_info


def get_videos_info(repo_id: str, local_dir: Path, video_keys: list[str]) -> dict:
    hub_api = HfApi()
    videos_info_dict = {"videos_path": VIDEO_PATH}
    for vid_key in video_keys:
        # Assumes first episode
        video_path = VIDEO_PATH.format(video_key=vid_key, episode_index=0)
        video_path = hub_api.hf_hub_download(
            repo_id=repo_id, repo_type="dataset", local_dir=local_dir, filename=video_path
        )
        videos_info_dict[vid_key] = _get_video_info(video_path)

    return videos_info_dict


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")


def get_video_shapes(videos_info: dict, video_keys: list) -> dict:
    video_shapes = {}
    for img_key in video_keys:
        channels = get_video_pixel_channels(videos_info[img_key]["video.pix_fmt"])
        video_shapes[img_key] = {
            "width": videos_info[img_key]["video.width"],
            "height": videos_info[img_key]["video.height"],
            "channels": channels,
        }

    return video_shapes


def get_image_shapes(dataset: Dataset, image_keys: list) -> dict:
    image_shapes = {}
    for img_key in image_keys:
        image = dataset[0][img_key]  # Assuming first row
        channels = get_image_pixel_channels(image)
        image_shapes[img_key] = {
            "width": image.width,
            "height": image.height,
            "channels": channels,
        }

    return image_shapes


def get_generic_motor_names(sequence_shapes: dict) -> dict:
    return {key: [f"motor_{i}" for i in range(length)] for key, length in sequence_shapes.items()}


def convert_dataset(
    repo_id: str,
    local_dir: Path,
    single_task: str | None = None,
    tasks_path: Path | None = None,
    tasks_col: Path | None = None,
    robot_config: dict | None = None,
):
    v1 = get_hub_safe_version(repo_id, V16, enforce_v2=False)
    v1x_dir = local_dir / V16 / repo_id
    v20_dir = local_dir / V20 / repo_id
    v1x_dir.mkdir(parents=True, exist_ok=True)
    v20_dir.mkdir(parents=True, exist_ok=True)

    hub_api = HfApi()
    hub_api.snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=v1, local_dir=v1x_dir, ignore_patterns="videos/"
    )

    metadata_v1 = load_json(v1x_dir / "meta_data" / "info.json")
    dataset = datasets.load_dataset("parquet", data_dir=v1x_dir / "data", split="train")
    keys = get_keys(dataset)

    # Episodes
    episode_indices = sorted(dataset.unique("episode_index"))
    total_episodes = len(episode_indices)
    assert episode_indices == list(range(total_episodes))

    # Tasks
    if single_task:
        tasks_by_episodes = {ep_idx: single_task for ep_idx in episode_indices}
        dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
        tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    elif tasks_path:
        tasks_by_episodes = load_json(tasks_path)
        tasks_by_episodes = {int(ep_idx): task for ep_idx, task in tasks_by_episodes.items()}
        # tasks = list(set(tasks_by_episodes.values()))
        dataset, tasks = add_task_index_by_episodes(dataset, tasks_by_episodes)
        tasks_by_episodes = {ep_idx: [task] for ep_idx, task in tasks_by_episodes.items()}
    elif tasks_col:
        dataset, tasks, tasks_by_episodes = add_task_index_from_tasks_col(dataset, tasks_col)
    else:
        raise ValueError

    assert set(tasks) == {task for ep_tasks in tasks_by_episodes.values() for task in ep_tasks}
    task_json = [{"task_index": task_idx, "task": task} for task_idx, task in enumerate(tasks)]
    write_json(task_json, v20_dir / "meta" / "tasks.json")

    # Split data into 1 parquet file by episode
    episode_lengths = split_parquet_by_episodes(dataset, keys, total_episodes, episode_indices, v20_dir)

    # Shapes
    sequence_shapes = {key: dataset.features[key].length for key in keys["sequence"]}
    image_shapes = get_image_shapes(dataset, keys["image"]) if len(keys["image"]) > 0 else {}
    if len(keys["video"]) > 0:
        assert metadata_v1.get("video", False)
        videos_info = get_videos_info(repo_id, v1x_dir, video_keys=keys["video"])
        video_shapes = get_video_shapes(videos_info, keys["video"])
        for img_key in keys["video"]:
            assert math.isclose(videos_info[img_key]["video.fps"], metadata_v1["fps"], rel_tol=1e-3)
            if "encoding" in metadata_v1:
                assert videos_info[img_key]["video.pix_fmt"] == metadata_v1["encoding"]["pix_fmt"]
    else:
        assert len(keys["video"]) == 0
        videos_info = None
        video_shapes = {}

    # Names
    if robot_config is not None:
        robot_type = robot_config["robot_type"]
        names = robot_config["names"]
        repo_tags = [robot_type]
    else:
        robot_type = "unknown"
        names = get_generic_motor_names(sequence_shapes)
        repo_tags = None

    assert set(names) == set(keys["sequence"])
    for key in sequence_shapes:
        assert len(names[key]) == sequence_shapes[key]

    # Episodes
    episodes = [
        {"episode_index": ep_idx, "tasks": [tasks_by_episodes[ep_idx]], "length": episode_lengths[ep_idx]}
        for ep_idx in episode_indices
    ]
    write_json(episodes, v20_dir / "meta" / "episodes.json")

    # Assemble metadata v2.0
    metadata_v2_0 = {
        "codebase_version": V20,
        "data_path": PARQUET_PATH,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": len(dataset),
        "total_tasks": len(tasks),
        "fps": metadata_v1["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "keys": keys["sequence"],
        "video_keys": keys["video"],
        "image_keys": keys["image"],
        "shapes": {**sequence_shapes, **video_shapes, **image_shapes},
        "names": names,
        "videos": videos_info,
    }
    write_json(metadata_v2_0, v20_dir / "meta" / "info.json")
    convert_stats_to_json(v1x_dir / "meta_data", v20_dir / "meta")

    #### TODO: delete
    repo_id = f"aliberts/{repo_id.split('/')[1]}"
    # if hub_api.repo_exists(repo_id=repo_id, repo_type="dataset"):
    #     hub_api.delete_repo(repo_id=repo_id, repo_type="dataset")
    hub_api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    ####

    with contextlib.suppress(EntryNotFoundError):
        hub_api.delete_folder(repo_id=repo_id, path_in_repo="data", repo_type="dataset", revision="main")

    with contextlib.suppress(EntryNotFoundError):
        hub_api.delete_folder(repo_id=repo_id, path_in_repo="meta_data", repo_type="dataset", revision="main")

    hub_api.upload_folder(
        repo_id=repo_id,
        path_in_repo="data",
        folder_path=v20_dir / "data",
        repo_type="dataset",
        revision="main",
    )
    hub_api.upload_folder(
        repo_id=repo_id,
        path_in_repo="meta",
        folder_path=v20_dir / "meta",
        repo_type="dataset",
        revision="main",
    )

    card_text = f"[meta/info.json](meta/info.json)\n```json\n{json.dumps(metadata_v2_0, indent=4)}\n```"
    push_dataset_card_to_hub(repo_id=repo_id, revision="main", tags=repo_tags, text=card_text)
    create_branch(repo_id=repo_id, branch=V20, repo_type="dataset")

    # TODO:
    # - [X] Add shapes
    # - [X] Add keys
    # - [X] Add paths
    # - [X] convert stats.json
    # - [X] Add task.json
    # - [X] Add names
    # - [X] Add robot_type
    # - [X] Add splits
    # - [X] Push properly to branch v2.0 and delete v1.6 stuff from that branch
    # - [X] Handle multitask datasets
    # - [/] Add sanity checks (encoding, shapes)


def main():
    parser = argparse.ArgumentParser()
    task_args = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset (e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    )
    task_args.add_argument(
        "--single-task",
        type=str,
        help="A short but accurate description of the single task performed in the dataset.",
    )
    task_args.add_argument(
        "--tasks-col",
        type=str,
        help="The name of the column containing language instructions",
    )
    task_args.add_argument(
        "--tasks-path",
        type=Path,
        help="The path to a .json file containing one language instruction for each episode_index",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="Path to the robot's config yaml the dataset during conversion.",
    )
    parser.add_argument(
        "--robot-overrides",
        type=str,
        nargs="*",
        help="Any key=value arguments to override the robot config values (use dots for.nested=overrides)",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="Local directory to store the dataset during conversion. Defaults to /tmp/{repo_id}",
    )

    args = parser.parse_args()
    if not args.local_dir:
        args.local_dir = Path(f"/tmp/{args.repo_id}")

    robot_config = parse_robot_config(args.robot_config, args.robot_overrides) if args.robot_config else None
    del args.robot_config, args.robot_overrides

    convert_dataset(**vars(args), robot_config=robot_config)


if __name__ == "__main__":
    from time import sleep

    sleep(1)
    main()