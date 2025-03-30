# HSE Reinforcement Learning Course Project

## Contributors

- Danil Sheshenya
- Temirkhan Zimanov

## Overview

In this project, we applied Group Relative Policy Optimization (GRPO), a technique proposed in the paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), to enhance the performance of compact language models on the letter-counting task. Specifically, our experiments involved fine-tuning the `Qwen2.5-0.5B-Instruct` and `Qwen2.5-1.5B-Instruct` models. A few sample prompts and correct answers are shown below.

| Prompt | Correct Answer |
| ------------- | ------------- |
| "Count the number of letters 'r' in the word 'strawberry'." | 3 |
| "Calculate the occurrences of 'z' in the word 'rizz'." | 2 |

## Installing dependencies

To install dependencies, run the following commands:

```bash
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1
```

## Preparing dataset

To prepare a dataset for training and testing, run the provided script:

```bash
python utils/prepare-dataset.py
```

This script generates a synthetic dataset specifically designed for the task of counting occurrences of a given letter within words. It leverages common English words from the Brown corpus (NLTK), selects random letters, and applies various prompt templates to simulate diverse input scenarios. The resulting dataset is stored in Parquet format (`train.parquet` and `test.parquet`) in the `data` directory, ready for model training and evaluation.

## Training

To train a model, execute the following command using Accelerate with DeepSpeed on an 8x H100 GPU node:

```bash
accelerate launch --config_file src/configs/deepspeed.yaml train.py trainer.run_name="<run_name>"
```

The training script (`train.py`) uses Hydra for configuration management and TRL's `GRPOTrainer` for reinforcement learning with the Group Relative Policy Optimization (GRPO) algorithm. The model is trained to provide the final numerical answer within `\boxed{}`. The reward during training is calculated based on accuracy, comparing the numerical answers extracted from the boxed outputs against the correct answers. Training configurations, model initialization, and dataset setup are handled via structured YAML files located in `src/configs`. WandB integration is included for experiment tracking. The checkpoint is saved to `saved/<run_name>` directory.

## Evaluation

To evaluate a model, run the inference script as follows:

```bash
python inference.py model.path="<path_or_name_of_the_model>"
```

During evaluation, the model is instructed to provide the final numerical answer within `\boxed{}` in its generation. The evaluation script (`inference.py`) loads the specified model and assesses its accuracy on the evaluation dataset through batched inference. Numerical answers are extracted directly from the model's boxed outputs and compared to ground truth values. The script leverages vLLM for efficient inference and reports the accuracy as the proportion of correctly predicted counts for the letter-counting task.

## Results

The table below summarizes the results of our experiments, highlighting the accuracy improvements achieved by training the Qwen models using GRPO:

| Model | Baseline Accuracy | Post-Training Accuracy | Training Time |
| ------------- | ------------- | ------------- | ------------- |
| Random | 0.300 | - | - |
| `Qwen2.5-0.5B-Instruct` | 0.071 | 0.332 | ~1 hour |
| `Qwen2.5-1.5B-Instruct` | 0.230 | 0.512 | ~2 hours |

The bigger model significantly outperformed the random baseline, demonstrating the effectiveness of GRPO training for enhancing mathematical reasoning in small-scale language models. The 1.5B-parameter model achieved more than double its initial accuracy within two hours of training. 

The following graph shows the training progress tracked using Weights and Biases (WandB):

![Training Progress](images/wandb_training_graph.png)

Below are sample generations given the prompt "Count the number of times 'z' appears in the word 'rizz'."

`Qwen2.5-0.5B-Instruct` **before** training:

```
To calculate the number of times 'z' appears in the word 'rizz', we can follow these steps:

1. Count each occurrence:
   - In "rizz", there is one 'z'.

2. Confirm that this count is accurate.

Therefore, the number of times 'z' appears in the word 'rizz' is **1**.
```

`Qwen2.5-0.5B-Instruct` **after** training:

```
To determine how many times the letter 'z' appears in the word "rizz", we can follow these steps:

1. **Identify the Word**: The word is "rizz".
2. **Count Each Character**:
   - 'r': Appears once.
   - 'i': Appears once.
   - 'z': Appears twice.

3. **Summarize**: We see that 'z' appears twice in the word "rizz".

Therefore, the number of times 'z' appears in the word "rizz" is \(\boxed{2}\).
```

`Qwen2.5-1.5B-Instruct` **before** training:

```
To calculate the occurrences of the letter 'z' in the word "rizz", we can follow these steps:

1. Identify each character in the word.
2. Count how many times the letter 'z' appears.

Let's break it down:
- The word is "rizz".
- We need to check for the letter 'z'.

In this case, there is only one 'z' in the word "rizz".

Therefore, the number of occurrences of 'z' in the word 'rizz' is 1.

The final answer is: \(\boxed{1}\)
```

`Qwen2.5-1.5B-Instruct` **after** training:

```
To calculate the occurrences of the letter 'z' in the word "rizz", we can follow these steps:

1. Identify each character in the word.
2. Count how many times the letter 'z' appears.

Let's go through this process step-by-step for the word "rizz":

1. The characters in the word "rizz" are: r, i, z, z.
2. We count the number of 'z's:
   - The first character is 'r'.
   - The second character is 'i'.
   - The third character is 'z'.
   - The fourth character is 'z'.

So, there are 2 occurrences of the letter 'z' in the word "rizz".

Therefore, the final answer is \(\boxed{2}\).
```

Both models are able to correctly answer the prompt after training.