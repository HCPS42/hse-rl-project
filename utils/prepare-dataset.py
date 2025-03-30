import os
import argparse
import json
from string import ascii_lowercase
from tqdm import tqdm

from datasets import Dataset
import numpy as np

import nltk
from nltk.corpus import words

def get_words() -> list[str]:
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
    return words.words()


def random_letter() -> str:
    return np.random.choice(list(ascii_lowercase)).item()

def generate_batch(words, size) -> tuple[list[str], list[str], list[int]]:
    words_batch = np.random.choice(list(words), size=size).tolist()
    letters_batch = np.random.choice(list(ascii_lowercase), size=size).tolist()
    counts_batch = [word.count(letter) for word, letter in zip(words_batch, letters_batch)]

    return words_batch, letters_batch, counts_batch


def build_dataset(
    words: list[str],
    num_samples: int,
    prompt_templates: list[str],
    distribution: list[float] = [0.05, 0.2, 0.4, 0.3, 0.05],
    batch_size: int = 10_000_000,
) -> tuple[Dataset, list[str]]:

    distribution = np.array(distribution) * num_samples
    distribution = np.round(distribution).astype(int).tolist() + [0] * 10
    prompt_templates = np.random.choice(prompt_templates, size=num_samples).tolist()
    words_batch, letters_batch, counts_batch = generate_batch(words, batch_size)
    idx = 0

    dataset = []
    pbar = tqdm(prompt_templates, total=num_samples, desc="Building dataset ...")
    for template in pbar:
        while distribution[counts_batch[idx]] == 0:
            idx += 1

        word = words_batch[idx]
        char = letters_batch[idx]
        count = counts_batch[idx]

        distribution[count] -= 1
        idx += 1

        dataset.append({
            "prompt": [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. You should think step by step. Return the final answer within \\boxed{}."},
                {"role": "user", "content": template.format(word=word, char=char)}
            ],
            "word": word,
            "char": char,
            "count": count
        })
        pbar.set_description(f"Building dataset ... (buffer: {idx} / {batch_size})")
    
    dataset = Dataset.from_list(dataset)
    dataset = dataset.shuffle()

    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-train-samples", type=int, default=10000)
    parser.add_argument("--num-test-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)

    with open("utils/prompt-templates.json", "r") as f:
        prompt_templates = json.load(f)

    words = get_words()
    np.random.shuffle(words)
    split_idx = len(words) // 2

    train_dataset = build_dataset(
        words[:split_idx],
        args.num_train_samples,
        prompt_templates
    )
    test_dataset = build_dataset(
        words[split_idx:],
        args.num_test_samples,
        prompt_templates
    )

    os.makedirs("data", exist_ok=True)
    train_dataset.to_parquet("data/train.parquet")
    test_dataset.to_parquet("data/test.parquet")


if __name__ == "__main__":
    main()