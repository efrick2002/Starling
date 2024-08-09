from transformers import AutoTokenizer, AutoModelForCausalLM
from torch import nn
import torch

# model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5")
tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")


class VModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5").model
        self.v_head = nn.Linear(4096, 1, bias=False)

    def forward(self, *args, **kwargs):
        return self.v_head(self.base_model(*args, **kwargs)[0]).squeeze(-1)


model = VModel()
tokenizer.pad_token = tokenizer.eos_token
input = tokenizer(["Hello, my dog is cute", "How are you"], padding=True, return_tensors="pt")
out = model(input_ids=input["input_ids"], attention_mask=input["attention_mask"])
print(out)
print(out.shape)

torch.save(model.state_dict(), "v_model.pt")

new_model = VModel()
print(new_model(input_ids=input["input_ids"], attention_mask=input["attention_mask"]))

new_model.load_state_dict(torch.load("v_model.pt"))
new_out = new_model(input_ids=input["input_ids"], attention_mask=input["attention_mask"])


assert (out == new_out).all()
