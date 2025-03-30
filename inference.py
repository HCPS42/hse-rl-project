import hydra
from hydra.utils import instantiate

import torch
from torch.utils.data import DataLoader
from vllm import SamplingParams

from src.utils import get_reward_fn


@hydra.main(version_base=None, config_path="src/configs", config_name="inference.yaml")
def main(config):
    model = instantiate(config.model)
    tokenizer = model.get_tokenizer()
    dataset = instantiate(config.dataset)
    dataloader = DataLoader(torch.arange(len(dataset)), batch_size=config.batch_size, shuffle=False)

    if config.sampling.top_k is None:
        config.sampling.top_k = -1
    if config.sampling.min_p is None:
        config.sampling.min_p = 0

    sampling_params = SamplingParams(
        **config.sampling,
        max_tokens=2048,
        skip_special_tokens=True,
    )

    accuracy_fn = get_reward_fn("binary")

    hits = 0
    for batch_idxs in dataloader:
        prompts = [dataset[idx.item()]["prompt"] for idx in batch_idxs]
        words = [dataset[idx.item()]["word"] for idx in batch_idxs]
        chars = [dataset[idx.item()]["char"] for idx in batch_idxs]
        counts = [dataset[idx.item()]["count"] for idx in batch_idxs]

        processed_prompts = [
            tokenizer.apply_chat_template(
                conversation=prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in prompts
        ]

        responses = model.generate(
            prompts=processed_prompts,
            sampling_params=sampling_params,
        )
        completions = [[{"content": response.outputs[0].text}] for response in responses]
        rewards = accuracy_fn(prompts, completions, words, chars, counts)
        hits += sum(rewards)

    print("*" * 20) 
    print("Model: ", config.model.path)
    print(f"Accuracy: {hits / len(dataset)}")

if __name__ == "__main__":
    main()

