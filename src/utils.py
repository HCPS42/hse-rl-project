import re

def binary_reward_fn(prompts, completions, word, char, count):
    rewards = []
    for completion, true_count in zip(completions, count):
        numbers = [int(num) for num in re.findall(r'\d+', completion)]

        if len(numbers) == 1:
            reward = (numbers[0] == true_count)
        else:
            reward = 0
        rewards.append(float(reward))
    
    return rewards

def abs_reward_fn(prompts, completions, word, char, count):
    rewards = []
    for completion, true_count in zip(completions, count):
        numbers = [int(num) for num in re.findall(r'\d+', completion)]

        if len(numbers) == 1:
            reward = -abs(numbers[0] - true_count)
        else:
            reward = -10
        rewards.append(float(reward))
    
    return rewards

def get_reward_fn(name):
    if name == "binary":
        return binary_reward_fn
    elif name == "abs":
        return abs_reward_fn
    else:
        raise ValueError(f"Invalid reward function: {name}")