# Setup

## Install dependencies

```bash
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1
```
## Prepare data

```bash
python utils/prepare-dataset.py
```

# Run Training

```bash
accelerate launch --config_file src/configs/deepspeed.yaml train.py trainer.run_name="<run_name>"
```

# Run Evaluation

```bash
python inference.py model.path="<path_or_name_of_the_model>"
```

