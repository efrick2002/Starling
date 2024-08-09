import json
import math
import os
import sys
import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import from_openchat_to_llama, from_list_to_openchat

# os.environ["HOME"] = "/scratch/banghua"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,7"
access_token = "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"


def create_reward_fn():  # noqa:  C901
    class GPTRewardModel(nn.Module):
        def __init__(self, model_path, eos_token_id, alpha):
            super().__init__()
            if "mistral" in model_path or "Llama" in model_path:
                model = AutoModelForCausalLM.from_pretrained(model_path, token=access_token)
                self.transformer = model.model
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path)
                self.transformer = model.gpt_neox
            self.config = model.config
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
            self.model = model
            # self.transformer = model.model
            self.alpha = alpha
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(self.tokenizer.eos_token)
            self.tokenizer.eos_token_id = eos_token_id
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

    reward_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token)
    reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf", reward_tokenizer.eos_token_id, 0.5)
    reward_tokenizer = reward_model.tokenizer
    print(reward_tokenizer.pad_token)
    reward_tokenizer.truncation_side = "left"

    directory = snapshot_download("banghua/n_rm")
    for fpath in os.listdir(directory):
        if fpath.endswith(".pt") or fpath.endswith(".bin"):
            checkpoint = os.path.join(directory, fpath)
            break

    reward_model.load_state_dict(torch.load(checkpoint), strict=False)
    reward_model.eval()
    reward_model.requires_grad_(False)
    reward_device = torch.cuda.device_count() - 1
    reward_model = reward_model.half().to(reward_device)
    reward_batch_size = 24

    def get_reward(samples):
        input_ids = []
        attention_masks = []
        encodings_dict = reward_tokenizer(
            samples,
            truncation=True,
            max_length=4096,
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
        samples = [from_openchat_to_llama(s) for s in samples]
        rewards = get_reward(samples)
        return rewards

    return reward_fn


def main(hparams={}):
    test_format_samples = [{"conversations": ["Hello"]}, {"conversations": ["Hello", "Hi", "How are you today?"]}]
    reward_tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    # assert reward_tokenizer(from_list_to_openchat(test_format_samples[0])["prompt"]).input_ids == [
    #     1,
    #     420,
    #     6316,
    #     28781,
    #     3198,
    #     3123,
    #     1247,
    #     28747,
    #     22557,
    #     32000,
    #     420,
    #     6316,
    #     28781,
    #     3198,
    #     3123,
    #     21631,
    #     28747,
    # ]
    # assert reward_tokenizer(from_list_to_openchat(test_format_samples[1])["prompt"]).input_ids == [
    #     1,
    #     420,
    #     6316,
    #     28781,
    #     3198,
    #     3123,
    #     1247,
    #     28747,
    #     22557,
    #     32000,
    #     420,
    #     6316,
    #     28781,
    #     3198,
    #     3123,
    #     21631,
    #     28747,
    #     15359,
    #     32000,
    #     420,
    #     6316,
    #     28781,
    #     3198,
    #     3123,
    #     1247,
    #     28747,
    #     1602,
    #     460,
    #     368,
    #     3154,
    #     28804,
    #     32000,
    #     420,
    #     6316,
    #     28781,
    #     3198,
    #     3123,
    #     21631,
    #     28747,
    # ]

    test_openchat = (
        "GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi <|end_of_turn|> GPT4 Correct User: How's it going?<|end_of_turn|>GPT4 Correct Assistant: It's good.<|end_of_turn|>"
    )
    test_llama = "[INST] Hello [/INST] Hi</s> [INST] How's it going? [/INST] It's good.</s>"
    assert from_openchat_to_llama(test_openchat) == test_llama

    reward_fn = create_reward_fn()
    print(
        reward_fn(
            samples=[test_openchat, test_openchat, test_openchat],
            prompts=None,
            outputs=None,
            addi=3,
        )
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
