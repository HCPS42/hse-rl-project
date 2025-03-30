from datasets import Dataset

def get_dataset(path: str) -> Dataset:
    return Dataset.from_parquet(path)
