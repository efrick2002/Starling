from os import environ
from os.path import abspath, dirname, join

ROOT_DIR = dirname(abspath(__file__))
PARENT_DIR = abspath(join(ROOT_DIR, ".."))
CACHE_DIR = join(PARENT_DIR, "cache")

environ["HF_DATASETS_CACHE"] = CACHE_DIR
environ["TRANSFORMERS_CACHE"] = CACHE_DIR
environ["PYTHONUNBUFFERED"] = "true"

import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoTokenizer,
)
from utils import reward_loss_7wise
from model_registry import MODEL_REGISTRY


def create_reward_model_tokenizer(
    model_name: str, use_cls=False
) -> PreTrainedTokenizerBase:
    """Creates the tokenizer for training. Handles the pad token and cls token logic.

    Args:
        model_name (str): HF model name
        use_cls (bool, optional): Whether to add a cls token. Defaults to False.

    Raises:
        NotImplementedError: If there is no pad token or unk token, the pad token is not defined which will cause an error.

    Returns:
        PreTrainedTokenizerBase: Tokenizer for training.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.truncation_side = "left"

    if use_cls:
        tokenizer.add_special_tokens({"cls_token": "<|cls|>"})

    if tokenizer.pad_token_id is not None:
        return tokenizer
    elif tokenizer.unk_token_id is not None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
        return tokenizer
    else:
        raise NotImplementedError(
            "Cannot find a good token to use as a pad token, please implement another case in this function."
        )


def create_reward_model(
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    use_cls=False,
    k=7,
    zero_sum_penalty=0.0,
    inference=False,
) -> PreTrainedModel:
    """Creates the transformer class for training.  Handles the logic for picking the correct transformer model.  It also handles the legacy class structure.

    Args:
        model_name (str): HF model name.
        tokenizer (PreTrainedTokenizerBase): tokenizer created by create_reward_model_tokenizer.
        use_cls (bool, optional): Whether to use a cls token for getting the reward. Defaults to False.
        k (int, optional): How many choices per prompt in the dataset. Defaults to 7.
        zero_sum_penalty (float, optional): Helps the model train rewards that are centered around 0. Defaults to 0.0.
        inference (bool, optional): If the model is for inferencing only. Defaults to False.

    Raises:
        NotImplementedError: If the model is not in the model registry.

    Returns:
        PreTrainedModel: Returns the model from which .from_pretrained can be called.
    """

    if model_name not in MODEL_REGISTRY:
        raise NotImplementedError(
            "Please add this model to the MODEL_REGISTRY so the correct transformer modules are used."
        )

    transformer_model_class, pretrained_model_class = MODEL_REGISTRY[model_name]

    class CustomAutoPreTrainedModel(pretrained_model_class):
        """Defines the appropriate pretrained class for the given model name.  This is done so that the value head init scheme is correct."""

        def _init_weights(self, module):
            std = self.config.initializer_range
            if isinstance(module, nn.Linear):
                module.reset_parameters()
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    class CustomAutoModelForSequenceClassification(CustomAutoPreTrainedModel):
        """Defines the reward model class using the correct transformer module."""

        def __init__(self, config):
            super().__init__(config)

            self.transformer = transformer_model_class(config)

            self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
            self.PAD_ID = tokenizer.pad_token_id
            self.CLS_ID = tokenizer.cls_token_id
            self.K = k
            self.zero_sum_penalty = zero_sum_penalty

            # Initialize weights and apply final processing
            self.post_init()

        def freeze_transformer(self):
            for param in self.transformer.parameters():
                param.requires_grad = False

        def get_device(self):
            return self.transformer.device

        def debug_init(self):
            mu = self.v_head.weight.float().mean()
            sd = self.v_head.weight.float().std()

            print(self.v_head.weight.shape)
            print("mean ", mu)
            print("std ", sd)
            print("weights ", self.v_head.weight)

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
            self.transformer.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            return

        def get_input_embeddings(self):
            return self.transformer.embed_tokens

        def set_input_embeddings(self, value):
            self.transformer.embed_tokens = value

        def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            mc_token_ids=None,
            labels=None,
            return_dict=False,
            output_attentions=False,
            output_hidden_states=False,
        ):

            # Split the inputs and rewards into two parts, chosen and rejected
            bs = int(input_ids.shape[0] / self.K)
            if inference:
                transformer_outputs = self.transformer(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_hidden_states=True,
                )
                hidden_states = transformer_outputs.hidden_states[-1]
                scores = []
                rewards = self.v_head(hidden_states).squeeze(-1)
                bs = int(input_ids.shape[0])

                for i in range(bs):
                    if use_cls:
                        c_inds = (input_ids[i] == self.CLS_ID).nonzero()
                        assert (
                            len(c_inds) > 0
                        ), "There should always be a CLS token but it was not found."
                        c_ind = c_inds[0].item()
                        scores.append(rewards[i, c_ind])
                    else:
                        c_inds = (input_ids[i] == self.PAD_ID).nonzero()
                        c_ind = (
                            c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
                        )
                        scores.append(rewards[i, c_ind - 1])
                scores = torch.stack(scores)
                return {"scores": scores}

            assert len(input_ids.shape) == 2, input_ids.shape

            # print("Start Get Transformer Outputs")
            transformer_outputs = self.transformer(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )

            hidden_states = transformer_outputs.hidden_states[-1]

            rewards = self.v_head(hidden_states).squeeze(-1)

            loss = 0
            scores_list = []
            for i in range(bs):
                scores = []
                for j in range(self.K):
                    if use_cls:
                        # Find CLS token and use reward on that index
                        c_inds = (input_ids[i * self.K + j] == self.CLS_ID).nonzero()
                        assert (
                            len(c_inds) > 0
                        ), "There should always be a CLS token but it was not found."
                        c_ind = c_inds[0].item()
                        scores.append(rewards[i * self.K + j][c_ind])
                    else:
                        # Check if there is any padding otherwise take length of sequence
                        c_inds = (input_ids[i * self.K + j] == self.PAD_ID).nonzero()
                        c_ind = (
                            c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
                        )
                        scores.append(rewards[i * self.K + j][c_ind - 1])

                scores_list.append(scores)
                scores = torch.stack(scores)
                loss += reward_loss_7wise(scores)

                if self.zero_sum_penalty > 0:
                    score_sum = scores.sum()
                    score_sum_squared = score_sum.square()
                    loss += self.zero_sum_penalty * score_sum_squared

            loss = loss / bs / (self.K - 1)
            return {"loss": loss, "scores_list": scores_list}

    return CustomAutoModelForSequenceClassification

