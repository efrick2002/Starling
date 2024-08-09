# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import from_openchat_to_llama, from_list_to_openchat
from datasets import load_dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5")
dataset = load_dataset("ThWu/rlhf_cleaned_prompt", split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)
dataset = dataset.map(from_list_to_openchat)

gen_kwargs = dict(
    max_new_tokens=2048,
    top_k=0,
    top_p=1.0,
    do_sample=True,
)

input = tokenizer(dataset["train"][:2]["prompt"], truncation=True, padding=False, max_length=2048, add_special_tokens=False)
tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.pad_token)
input = tokenizer.pad(input, return_tensors="pt")
print(input["input_ids"])
print(tokenizer.batch_decode(input["input_ids"]))
gen_out = model.generate(input_ids=input["input_ids"], attention_mask=input["attention_mask"], **gen_kwargs)
print(gen_out)
print(tokenizer.batch_decode(gen_out))

"""
test reward head
"""
# print(model.eval())

# # MistralForCausalLM(
# #   (model): MistralModel(
# #     (embed_tokens): Embedding(32002, 4096)
# #     (layers): ModuleList(
# #       (0-31): 32 x MistralDecoderLayer(
# #         (self_attn): MistralAttention(
# #           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
# #           (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
# #           (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
# #           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
# #           (rotary_emb): MistralRotaryEmbedding()
# #         )
# #         (mlp): MistralMLP(
# #           (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
# #           (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
# #           (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
# #           (act_fn): SiLUActivation()
# #         )
# #         (input_layernorm): MistralRMSNorm()
# #         (post_attention_layernorm): MistralRMSNorm()
# #       )
# #     )
# #     (norm): MistralRMSNorm()
# #   )
# #   (lm_head): Linear(in_features=4096, out_features=32002, bias=False)
# # )

# input = tokenizer("How are you", return_tensors="pt")
# output = model(input_ids=input["input_ids"], attention_mask=input["attention_mask"], output_hidden_states=True)
# hidden_state = output.hidden_states[-1]
# hidden_state1 = model.model(input_ids=input["input_ids"], attention_mask=input["attention_mask"])[0]
# print(output[0])
# print(output.logits)
# print(hidden_state1)
# assert torch.eq(hidden_state, hidden_state1).all()

# lm_head = model.lm_head
# hidden_state.shape  # torch.Size([1, 4, 4096])
# output.logits.shape  # torch.Size([1, 4, 32002])
# assert torch.eq(lm_head(hidden_state), output.logits).all()
