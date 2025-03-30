import hydra
from hydra.utils import instantiate

from torch.utils.data import DataLoader
from vllm import SamplingParams

from src.utils import get_reward_fn


@hydra.main(version_base=None, config_path="src/configs", config_name="inference.yaml")
def main(config):
    model = instantiate(config.model)
    tokenizer = model.get_tokenizer()
    dataset = instantiate(config.dataset)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    sampling_params = SamplingParams(
        **config.sampling,
        skip_special_tokens=True,
    )

    accuracy_fn = get_reward_fn("binary")

    hits = 0
    for batch in dataloader:
        prompts = batch["prompt"]

        processed_prompts = [
            tokenizer.apply_chat_template(
                conversation=[
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in prompts
        ]

        responses = model.generate(
            prompts=processed_prompts,
            sampling_params=sampling_params,
        )
        completions = [response.output[0].text for response in responses]
        rewards = accuracy_fn(prompts, completions, batch["word"], batch["char"], batch["count"])
        hits += sum(rewards)
    
    print(f"Accuracy: {hits / len(dataset)}")

if __name__ == "__main__":
    main()

