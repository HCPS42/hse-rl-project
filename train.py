import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os

from trl import GRPOTrainer, GRPOConfig
import accelerate
import wandb

@hydra.main(version_base=None, config_path="src/configs", config_name="grpo.yaml")
def main(config):
    model, tokenizer = instantiate(config.model)
    dataset = instantiate(config.dataset)
    reward_fn = instantiate(config.reward_fn)

    config.trainer.output_dir = os.path.join(config.trainer.output_dir, config.trainer.run_name)
    training_args = GRPOConfig(**config.trainer)

    if os.environ.get("RANK", "0") == "0":
        wandb.init(
            project="rl-project",
            name=config.trainer.run_name,
            config=OmegaConf.to_container(config, resolve=True),
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
