import json
import math
import os
import sys
from itertools import islice

# import numpy as np
import torch

# import tritonclient.grpc as client_util
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType

# from tritonclient.utils import np_to_triton_dtype
from utils import from_openchat_to_llama, from_list_to_openchat

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    P3OConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

default_config = TRLConfig(
    train=TrainConfig(
        seq_length=2048,
        epochs=10000,
        total_steps=30000,
        batch_size=2,
        eval_batch_size=4,
        checkpoint_interval=500,
        eval_interval=500,
        pipeline="PromptPipeline",
        save_best=True,
        save_optimizer=False,
        trainer="AccelerateP3OTrainer",
        checkpoint_dir="checkpoints/p3o_hh",
    ),
    model=ModelConfig(
        model_path="openchat/openchat_3.5",
        # num_layers_unfrozen=7,
        peft_config=LoraConfig(
            r=8,
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=32,
            lora_dropout=0.1,
        ),
    ),
    tokenizer=TokenizerConfig(tokenizer_path="openchat/openchat_3.5", truncation_side="left"),
    optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=4e-8, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
    scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=4e-8)),
    method=P3OConfig(
        name="P3OConfig",
        num_responses_per_query=2,
        num_rollouts=32,
        chunk_size=4,
        p3o_epochs=2,
        kl_coef=0.01,
        cliprange=0.2,
        cliprange_ratio=10.0,
        scale_reward="running",
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs=dict(
            max_new_tokens=1024,
            top_k=0,
            top_p=1.0,
            do_sample=True,
        ),
        clip_tokenwise=False,
        avg_tokenwise=False,
        scale_q=False,
    ),
)
"""
old_version: 
clip_tokenwise=False,
avg_tokenwise=False,

new_version:
clip_tokenwise=True,
avg_tokenwise=True/False, doesn't matter

old_version(normalize loss tokenwise):
clip_tokenwise=False,
avg_tokenwise=True,
"""

# accelerate launch --num_processes 1 --config_file ../../configs/accelerate/zero2-bf16.yaml mistral_p3o.py


def create_reward_fn():  # noqa:  C901
    class GPTRewardModel(nn.Module):
        def __init__(self, model_path, eos_token_id, alpha):
            super().__init__()
            if "mistral" in model_path or "Llama" in model_path:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                self.transformer = model.model
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                self.transformer = model.gpt_neox
            self.config = model.config
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
            self.model = model
            self.alpha = alpha
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.unk_token
            self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]
            self.K = 7

        def get_device(self):
            return self.model.device

        def gradient_checkpointing_enable(self):
            self.model.gradient_checkpointing_enable()
            return

        def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            position_ids=None,
        ):
            """
            input_ids, attention_mask: torch.Size([bs, seq_len])
            return: scores: List[torch.Size([1])
            """
            bs = input_ids.shape[0]
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = transformer_outputs[0]
            scores = []
            rewards = self.v_head(hidden_states).squeeze(-1)
            for i in range(bs):
                c_inds = (input_ids[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
                scores.append(rewards[i, c_ind - 1])
            return scores

    reward_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf", reward_tokenizer.eos_token_id, 0.5)
    reward_tokenizer = reward_model.tokenizer
    print("Reward tokenizer pad token:", reward_tokenizer.pad_token)
    reward_tokenizer.truncation_side = "left"

    directory = snapshot_download("banghua/refine_rm")
    for fpath in os.listdir(directory):
        if fpath.endswith(".pt") or fpath.endswith("model.bin"):
            checkpoint = os.path.join(directory, fpath)
            break
    if os.environ.get("LOCAL_RANK", "0") == "0":
        reward_model.load_state_dict(torch.load(checkpoint), strict=False)
        reward_model.eval()

        reward_model.requires_grad_(False)
        reward_device = torch.cuda.device_count() - 1
        reward_model = reward_model.to(reward_device)
        reward_batch_size = 4

    def get_reward(samples):
        """samples: List[str]"""
        input_ids = []
        attention_masks = []
        encodings_dict = reward_tokenizer(
            samples,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        ).to(reward_device)
        input_ids = encodings_dict["input_ids"]
        attention_masks = encodings_dict["attention_mask"]

        mbs = reward_batch_size
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            rewards = reward_model(input_ids=input_ids[i * mbs : (i + 1) * mbs], attention_mask=attention_masks[i * mbs : (i + 1) * mbs])
            out.extend(rewards)
        return torch.hstack(out)

    def reward_fn(samples, prompts, **kwargs):
        samples = [from_openchat_to_llama(sample) for sample in samples]
        rewards = get_reward(samples)
        return rewards

    return reward_fn


def main(hparams={}):
    config = TRLConfig.update(default_config, hparams)
    dataset = load_dataset("ThWu/cleaned_prompt_r", split="train")
    dataset = dataset.train_test_split(test_size=0.001, seed=42)
    dataset = dataset.map(from_list_to_openchat)

    prompts = [{"prompt": x["prompt"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"]} for x in islice(dataset["test"], 100)]
    reward_fn = create_reward_fn()

    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["GPT4 Correct User:", "GPT4 Correct Assistant:"],
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
