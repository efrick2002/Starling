import gc
from dataclasses import dataclass
from copy import deepcopy
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import ModelOutput, BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from trlx.models.modeling_base import PreTrainedModelWrapper
import functools


def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "model.layers",
        "decoder.layers",
        "transformer.h",
        "transformer.blocks",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def freeze_bottom_causal_layers(model: transformers.PreTrainedModel, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)

    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
        hidden_layers_to_freeze += [model.get_input_embeddings(), model.get_output_embeddings()]
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
        hidden_layers_to_freeze += [model.get_input_embeddings()]
        if model.config.tie_word_embeddings:
            hidden_layers_to_freeze += [model.get_output_embeddings()]
    else:
        hidden_layers_to_freeze = []

    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


@dataclass
class CausalLMOutputWithValue(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    value: Optional[torch.FloatTensor] = None


def make_mistral_value_branch(base_model, num_value_layers_unfrozen):
    value_head = nn.Linear(4096, 1, bias=False)
    if num_value_layers_unfrozen == 0:
        return value_head
    value_branch = MistralModelBranch(base_model=base_model, num_layers_unfrozen=num_value_layers_unfrozen)
    value_branch.lm_head = value_head
    return value_branch


class MistralModelBranch(transformers.PreTrainedModel):
    """
    Take the last `num_layers_unfrozen` layers of the pretrained mistral model
    """

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        num_layers_unfrozen: int,
    ):
        super().__init__(base_model.config)
        self.padding_idx = base_model.model.config.pad_token_id
        self.vocab_size = base_model.model.config.vocab_size

        self.embed_tokens = deepcopy(base_model.model.embed_tokens)
        self.embed_tokens.requires_grad_(False)
        self.layers = deepcopy(base_model.model.layers[-num_layers_unfrozen:])
        self.norm = deepcopy(base_model.model.norm)
        self.lm_head = deepcopy(base_model.lm_head)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and hasattr(self.config, "_flash_attn_2_enabled") and self.config._flash_attn_2_enabled and past_key_values is not None:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        # hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            outputs = tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        else:
            outputs = BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MistralModelWithHydraHead(PreTrainedModelWrapper):
    """An `AutoModel` class wrapper for `transformers` causal models that have a
    language modeling head and a value head
    """

    def __init__(
        self,
        base_model: transformers.PreTrainedModel,
        peft_config=None,
        num_layers_unfrozen=0,
    ):
        super().__init__(base_model, peft_config=peft_config)
        self.num_layers_unfrozen = num_layers_unfrozen
        # self.v_head = make_mistral_value_branch(base_model, num_layers_unfrozen)
        self.v_head = None
        self.frozen_head = MistralModelBranch(base_model, num_layers_unfrozen)
        for param in self.frozen_head.parameters():
            param.requires_grad = False
        self.frozen_head = self.frozen_head.eval()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ignore_peft_adapter: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithValue]:
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_hidden_states": True,
            "return_dict": True,
        }

        outputs = self.base_model(**forward_kwargs)
        forward_kwargs["hidden_states"] = outputs["hidden_states"][-(self.num_layers_unfrozen + 1)]
        forward_kwargs.pop("return_dict", None)
        # value = self.v_head(**forward_kwargs).logits.squeeze(-1)
        value = None
        if not return_dict:
            outputs = (outputs.logits,) + outputs[1:] + (value,)
            return outputs

        return CausalLMOutputWithValue(**outputs, value=value)

    def forward_hydra(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        position_ids: Optional[List[torch.FloatTensor]] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.FloatTensor, CausalLMOutputWithValue]:
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_hidden_states": True,
            "return_dict": True,
        }
        outputs = self.forward(**forward_kwargs)
        forward_kwargs["hidden_states"] = outputs["hidden_states"][-(self.num_layers_unfrozen + 1)]
        hydra_outputs = self.frozen_head(**forward_kwargs)

        if not return_dict:
            return hydra_outputs.logits
        return hydra_outputs

    def generate(self, *args, **kwargs) -> Union[ModelOutput, torch.LongTensor]:
        return self.base_model.generate(*args, **kwargs)

    def state_dict(self, *args, heads_only=False, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        # state_dict = self.v_head.state_dict(*args, **dict(prefix="v_head.", **kwargs))
        state_dict = {}
        if not heads_only:
            state_dict = {
                **state_dict,
                **self.base_model.state_dict(*args, **dict(prefix="" if self.peft_type else "base_model.", **kwargs)),
            }

            if self.frozen_head:
                state_dict = {
                    **state_dict,
                    **self.frozen_head.state_dict(*args, **dict(prefix="frozen_head.", **kwargs)),
                }

        return state_dict

    def post_init(self, state_dict):
        """
        Load `state_dict` into the model. If peft was used to train the model,
        only the value head would be present in the loaded `state_dict`, so the
        loading has to be not strict. Also `frozen_head` will be recreated and
        loaded from the checkpoint, to comply with deepspeed checkpoint loading.
        """
        strict = not self.peft_type and any(k.startswith("base_model.") or k.startswith("v_head.") for k in state_dict)

        # if not self.peft_type and self.frozen_head is None:
        #     for k in state_dict:
        #         match = re.search(r"^frozen_head\..+\.(\d+)\.", k)
        #         if match:
        #             self.num_layers_unfrozen = max(self.num_layers_unfrozen, int(match.group(1)) + 1)

        #     config = self.base_model.config
        #     branch_class = hf_get_branch_class(config)
        #     self.frozen_head = branch_class(
        #         self.base_model,
        #         num_layers_unfrozen=self.num_layers_unfrozen,
        #     ).eval()

        self.load_state_dict(state_dict, strict=strict)
        del state_dict
        gc.collect()  # noqa: E702


tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
base_model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5")

input = tokenizer("Hello, my dog is cute", return_tensors="pt")
input["return_dict"] = True
hydra_model = MistralModelWithHydraHead(base_model=base_model, num_layers_unfrozen=5)

# MistralModelWithHydraHead(
#   (base_model): MistralForCausalLM(
#     (model): MistralModel(
#       (embed_tokens): Embedding(32002, 4096)
#       (layers): ModuleList(
#         (0-31): 32 x MistralDecoderLayer(
#           (self_attn): MistralAttention(
#             (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#             (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
#             (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
#             (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#             (rotary_emb): MistralRotaryEmbedding()
#           )
#           (mlp): MistralMLP(
#             (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
#             (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
#             (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
#             (act_fn): SiLUActivation()
#           )
#           (input_layernorm): MistralRMSNorm()
#           (post_attention_layernorm): MistralRMSNorm()
#         )
#       )
#       (norm): MistralRMSNorm()
#     )
#     (lm_head): Linear(in_features=4096, out_features=32002, bias=False)
#   )
#   (frozen_head): MistralModelBranch(
#     (embed_tokens): Embedding(32002, 4096)
#     (layers): ModuleList(
#       (0-4): 5 x MistralDecoderLayer(
#         (self_attn): MistralAttention(
#           (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
#           (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
#           (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
#           (rotary_emb): MistralRotaryEmbedding()
#         )
#         (mlp): MistralMLP(
#           (gate_proj): Linear(in_features=4096, out_features=14336, bias=False)
#           (up_proj): Linear(in_features=4096, out_features=14336, bias=False)
#           (down_proj): Linear(in_features=14336, out_features=4096, bias=False)
#           (act_fn): SiLUActivation()
#         )
#         (input_layernorm): MistralRMSNorm()
#         (post_attention_layernorm): MistralRMSNorm()
#       )
#     )
#     (norm): MistralRMSNorm()
#     (lm_head): Linear(in_features=4096, out_features=32002, bias=False)
#   )
# )

forward_output = hydra_model(**input)
hydra_output = hydra_model.forward_hydra(**input)

original_output = base_model(**input)

assert (forward_output.logits == original_output.logits).all()
assert (hydra_output.logits == original_output.logits).all()
print(forward_output.value)

# print(hydra_model.state_dict().keys())
# enumerate params and see if they require grad
print("before freezing")
for name, param in hydra_model.named_parameters():
    print(name, param.requires_grad)

print("after freezing")
freeze_bottom_causal_layers(hydra_model.base_model, 5)
for name, param in hydra_model.named_parameters():
    print(name, param.requires_grad)
