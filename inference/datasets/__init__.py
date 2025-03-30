from datasets.base import BaseDataset


dataset_map: dict[str, BaseDataset] = {
    "AIME-2025": BaseDataset("data/aime/aime-2025.csv"),
}

def get_dataset(dataset_name: str):
    return dataset_map[dataset_name]

__all__ = ["BaseDataset", "get_dataset", "dataset_map"]