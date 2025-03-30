from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM

def get_model(name: str):
    model = AutoModelForCausalLM.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

def get_vllm_model(name: str):
    model = LLM(
        model=name,
        dtype="half",
        gpu_memory_utilization=0.9,
        max_model_len=512,
        trust_remote_code=True,
        tensor_parallel_size=1,
    )
    return model
