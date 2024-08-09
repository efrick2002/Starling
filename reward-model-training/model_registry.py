from transformers import (
    LlamaModel,
    LlamaPreTrainedModel,
)

# Add models here for reward training support
MODEL_REGISTRY = {
    "HuggingFaceH4/tiny-random-LlamaForCausalLM": (LlamaModel, LlamaPreTrainedModel),
    "meta-llama/Llama-2-7b-chat-hf": (LlamaModel, LlamaPreTrainedModel),
    "01-ai/Yi-34B-Chat": (LlamaModel, LlamaPreTrainedModel),
}
