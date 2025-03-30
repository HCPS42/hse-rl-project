import hydra
from hydra.utils import instantiate
import os

from trl import GRPOTrainer, GRPOConfig

from src.utils import reward_fn

@hydra.main(version_base=None, config_path="src/configs", config_name="grpo.yaml")
def main(config):
    model, tokenizer = instantiate(config.model)
    dataset = instantiate(config.dataset)

    config.trainer.output_dir = os.path.join(config.trainer.output_dir, config.trainer.run_name)
    training_args = GRPOConfig(**config.trainer)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
