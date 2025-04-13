from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def merge_datasets(ds_repo_ids: list[str]):
    ds = LeRobotDataset(ds_repo_ids[0])
    for ds_repo_id in ds_repo_ids[1:]:
        ds.merge(LeRobotDataset(ds_repo_id))
    return ds
