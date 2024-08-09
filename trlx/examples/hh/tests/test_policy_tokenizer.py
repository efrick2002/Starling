import json
import sys

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import from_openchat_to_llama, from_list_to_openchat


tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
print(tokenizer)
dataset = load_dataset("ThWu/rlhf_cleaned_prompt", split="train[:200]")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset = dataset.map(from_list_to_openchat)
prompts = [{"prompt": x["prompt"]} for x in dataset["train"]]

if isinstance(prompts[0], dict):
    metadata = prompts
    prompts = [x.pop("prompt") for x in metadata]
else:
    metadata = [{}] * len(prompts)
# model_inputs = tokenizer(prompts, truncation=True, padding=False, max_length=4096, add_special_tokens=False)
print(tokenizer.eos_token)
print(tokenizer.eos_token_id)
input = tokenizer(dataset["train"][:2]["prompt"], truncation=True, padding=False, max_length=2048, add_special_tokens=False)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.pad_token)
input = tokenizer.pad(input, return_tensors="pt")
print(input["input_ids"])
print(tokenizer.batch_decode(input["input_ids"]))

# input = tokenizer(dataset["train"][:2]["prompt"], truncation=True, padding=False, max_length=2048, add_special_tokens=False)
# tokenizer.pad_token = tokenizer.eos_token
# print(tokenizer.pad_token)
# input = tokenizer.pad(input, return_tensors="pt")
# print(input["input_ids"])
# print(tokenizer.batch_decode(input["input_ids"]))
