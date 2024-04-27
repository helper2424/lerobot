"""
This script provides a utility for saving a dataset as safetensors files for the purpose of testing backward compatibility
when updating the data format. It uses the `PushtDataset` to create a DataLoader and saves selected frame from the
dataset into a corresponding safetensors file in a specified output directory.

If you know that your change will break backward compatibility, you should write a shortlived test by modifying
`tests/test_datasets.py::test_backward_compatibility` accordingly, and make sure this custom test pass. Your custom test
doesnt need to be merged into the `main` branch. Then you need to run this script and update the tests artifacts.

Example usage:
    `python tests/script/save_dataset_to_safetensors.py`
"""

import shutil
from pathlib import Path

from safetensors.torch import save_file

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def save_dataset_to_safetensors(output_dir, repo_id="lerobot/pusht"):
    data_dir = Path(output_dir) / repo_id

    if data_dir.exists():
        shutil.rmtree(data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    dataset = LeRobotDataset(repo_id)

    # save 2 first frames of first episode
    i = dataset.episode_data_index["from"][0].item()
    save_file(dataset[i], data_dir / f"frame_{i}.safetensors")
    save_file(dataset[i + 1], data_dir / f"frame_{i+1}.safetensors")

    # save 2 frames at the middle of first episode
    i = int((dataset.episode_data_index["to"][0].item() - dataset.episode_data_index["from"][0].item()) / 2)
    save_file(dataset[i], data_dir / f"frame_{i}.safetensors")
    save_file(dataset[i + 1], data_dir / f"frame_{i+1}.safetensors")

    # save 2 last frames of first episode
    i = dataset.episode_data_index["to"][0].item()
    save_file(dataset[i - 2], data_dir / f"frame_{i-2}.safetensors")
    save_file(dataset[i - 1], data_dir / f"frame_{i-1}.safetensors")

    # TODO(rcadene): Enable testing on second and last episode
    # We currently cant because our test dataset only contains the first episode

    # # save 2 first frames of second episode
    # i = dataset.episode_data_index["from"][1].item()
    # save_file(dataset[i], data_dir / f"frame_{i}.safetensors")
    # save_file(dataset[i+1], data_dir / f"frame_{i+1}.safetensors")

    # # save 2 last frames of second episode
    # i = dataset.episode_data_index["to"][1].item()
    # save_file(dataset[i-2], data_dir / f"frame_{i-2}.safetensors")
    # save_file(dataset[i-1], data_dir / f"frame_{i-1}.safetensors")

    # # save 2 last frames of last episode
    # i = dataset.episode_data_index["to"][-1].item()
    # save_file(dataset[i-2], data_dir / f"frame_{i-2}.safetensors")
    # save_file(dataset[i-1], data_dir / f"frame_{i-1}.safetensors")


if __name__ == "__main__":
    save_dataset_to_safetensors("tests/data/save_dataset_to_safetensors")