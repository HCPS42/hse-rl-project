params = {
    "model_path": "Qwen/Qwen2.5-1.5B",
    "devices": "0",
    "stages": {
        "slow": [
            {
                "max_tokens": 12288,
                "num_tries": 64,
                "temp": 1.0,
                "top_p": 0.9,
                "min_p": 0.05,
            }
        ]
    }
}