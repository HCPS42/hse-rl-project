from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
import torch

def get_model(name: str, attn_implementation: str):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    return model, tokenizer

def get_vllm_model(path: str):
    model = LLM(
        model=path,
        dtype="half",
        gpu_memory_utilization=0.9,
        max_model_len=512,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    return model
