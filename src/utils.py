import re


def reward_fn(prompts, completions, word, char, count, get_reward):
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for completion, true_count in zip(completion_contents, count):
        pattern = r'oxed{(.*?)}'
        numbers = [int(num) for num in re.findall(pattern, completion)]
        rewards.append(get_reward(numbers, true_count))
    
    return rewards

def binary_reward_fn(prompts, completions, word, char, count):
    def get_reward(numbers, true_count):
        if len(numbers) == 1:
            return float(numbers[0] == true_count)
        else:
            return 0

    return reward_fn(prompts, completions, word, char, count, get_reward)

def abs_reward_fn(prompts, completions, word, char, count):
    def get_reward(numbers, true_count):
        if len(numbers) == 1:
            return -abs(numbers[0] - true_count)
        else:
            return -10

    return reward_fn(prompts, completions, word, char, count, get_reward)

def get_reward_fn(name):
    if name == "binary":
        return binary_reward_fn
    elif name == "abs":
        return abs_reward_fn
    else:
        raise ValueError(f"Invalid reward function: {name}")