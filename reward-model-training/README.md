# Reward Model Training

This repository contains the code for training a reward model using PyTorch, and Transformers. It is designed to train a model to generate rewards based on input sequences, using a comparison-based training approach.

## Local Installation


We use venv for multinode training since it ensures all the paths are identical on both nodes.  If you want to use conda for single node that should work fine.

```bash
source setup.sh
```

Be wary of various libraries that might be missing on your local environment. You may need to apt-get a thing or two to get things working.

## Supported Models

The supported models can be found in [model_registry.py](./model_registry.py).  To add support to a new model, just add it to the registry.  Specifically, you need to define a mapping like this `"model_name" : (ModelTransformerBlock, ModelPretrainedModelClass)`.  You may need to import these classes from the transformers library if they are not already imported.  Note that before training, you may still need to define a chat template in the training config yaml.

### Adding a New Model

```python
from transformers import (
    LlamaModel,
    LlamaPreTrainedModel,
    CoolNewModel, # <-- Import your shiny new model.
    CoolNewPretrainedModel, # <-- Don't forget the pretained class too.
)

# Add models here for reward training support
MODEL_REGISTRY = {
    "HuggingFaceH4/tiny-random-LlamaForCausalLM": (LlamaModel, LlamaPreTrainedModel),
    "meta-llama/Llama-2-7b-chat-hf": (LlamaModel, LlamaPreTrainedModel),
    "01-ai/Yi-34B-Chat": (LlamaModel, LlamaPreTrainedModel),
    "CoolOrg/Cool-7T-Chat": (CoolNewModel, CoolNewPretrainedModel), # <-- Add your mapping to the dictionary.
}

```

## Training

The training code is located in [train.py](./train.py) which uses Transformers Trainer and Deepspeed Zero as the backbone infastructure. The following sections run through the steps needed to get training started.

### Configs
The yaml configs define the training params relevant to a training run.

These argument options are defined below. For more details on "Trainer" arguments, see the Huggingface [TrainingArguments](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments) documentation for more details.

The following arguments are REQUIRED:
```yaml
    model_name: Huggingface model name.
    learning_rate: Trainer learning rate.
    batch_size: Trainer batch size.
    gradient_accumulation_steps: Trainer gradient accumulation steps.
    data_path: Hugginface dataset name. 
    max_length: Training max token length.
    random_seed: Random seed, grows into a random tree if watered daily.
    deepspeed_config: Path to the Deepspeed config file. See './ds_config'.
```

The following arguments are OPTIONAL:
```yaml
    data_percentage: Training data percentage. Defaults to 100.
    num_train_epochs: Trainer number of epochs to train for. Defaults to 1.
    lr_scheduler_type: Trainer learning rate scheduler. Defaults to 'linear'
    warmup_ratio: Training warmup ratio. Defaults to 0.0.
    zero_sum_penalty: Zero sum penalty strength. Helps center the reward at 0, recommended for larger models where Zero3 is used. Defaults to 0.0 (no penalty).
    use_cls: Whether to add a <|cls|> token to standardize the token that recieve the reward value.
    chat_template: Path to the chat template for data formatting. Defaults to whatever is default to the tokenizer.  Check the tokenizer default chat template before you use it! (Some add system prompts).
```

### Deepspeed Configs

Deepspeed configs can be found in the [ds_configs](./ds_configs/) directory.  These are then passed into the training configs, under the key `deepspeed_config`.

More details on how to set up a deepspeed config can be found in [Deepspeed's Documentation](https://www.deepspeed.ai/docs/config-json/).

Note that during Multi-Node training, we need to set some obscure checkpointing configs if the checkpoints will not be saved to a shared file system.  Specifically this will need to be added:

```json
  "checkpoint": {
    "use_node_local_storage": true
  },
```

Other than that, the Deepspeed config set up is fairly standard.

### Chat Templates

To use a custom chat template, add it to the [chat_templates](./chat_templates/) directory.

### Argument Usage
Here are all the CLI args for training:



  `--config CONFIG, -c CONFIG` : Config file containing training arguments.

  `--model-path MODEL_PATH` : Huggingface model name.

  `--output-path OUTPUT_PATH, -op OUTPUT_PATH` : Use to specify a specific path in which to save the checkpoints (e.g. /scratch).

  `--checkpoint CHECKPOINT, -cp CHECKPOINT` : For restarting training at the checkpoint located at this inputted path.

  `--save-steps SAVE_STEPS` : How often to save a checkpoint.

  `--eval-steps EVAL_STEPS` : How often to run an evaluation step.

  `--save-on-each-node, -soen` : Use when checkpoints are not being saved to a shared node.

  `--save-total-limit SAVE_TOTAL_LIMIT, -stl SAVE_TOTAL_LIMIT` : The max amount of checkpoints to keep at any time.

  `--train-head-only` : Use this flag to freeze the transformer block and train the value head only.

  `--no-cached-dataset, -ncd` : Use this flag to avoid the cached dataset, helps avoid weird issues that might occur with caching.

  `--debug` :Debug flag, runs through a train step, eval step, checkpoint save, train step, end model save. Ignores save step, eval step, gradient accumulation args and sets them to 1, 1, and 2, respectively.

  `--local-rank LOCAL_RANK` : For deepspeed, do not set this yourself.

  `--local_rank LOCAL_RANK` : For deepspeed, do not set this yourself.

### Example Run Commands

```bash
# --debug run does 2 training steps, 2 evaluation steps, and 1 checkpoint save and 1 model save.  You can add --debug to any config and it will do this.
deepspeed --num_nodes=2 --num_gpus=8 --hostfile ./hostfile train.py --config training_configs/config_debug.yaml --debug

# setting --output-path puts all the training outputs in that directory.
deepspeed --num_nodes=2 --num_gpus=8 --hostfile ./hostfile train.py --config training_configs/config_yi_34b_chat.yaml --output-path /scratch/training_outputs

# We can restart from checkpoint with --checkpoint or -cp
deepspeed --num_nodes=2 --num_gpus=8 --hostfile ./hostfile train.py --config training_configs/config_yi_34b_chat.yaml --checkpoint path/to/checkpoint/dir

```

## Evaluating Reward Models

There are 2 groups of evaluations.
1. [The Internal Evalution Suite](./benchmarks/)
2. [Reward Bench](https://huggingface.co/spaces/allenai/reward-bench)

Both of these benchmarks can be run efficiently with [evaluate.py](./evaluate.py) which will save each benchmark in a json located in the root directory.  See the score section for visualizing the result.

### Internal Evaluations

The internal evaluations consists of Chatbot Arena battles (Human Preference), TruthfulQA triplets (Truth Preference), PKU Alignment pairs (Safety Preference), and handpicked adversarial verbosity pairs (Verbose Preference).

These evaluations are run by default.

### Reward Bench

Reward Bench can be run with the flag `--reward-bench`.

### Argument Usage
  `--model-path MODEL_PATH` : Huggingface model path.

  `--base-model BASE_MODEL` : Base model used to train the model, needs to be registered.

  `--chat-template CHAT_TEMPLATE` : Path to the chat template to use, if None, the default for the tokenizer is used.

  `--use-cls` : Flag for adding the cls token to the tokenizer and end of the each query.

  `--tensor-parallel-size TENSOR_PARALLEL_SIZE, -tp TENSOR_PARALLEL_SIZE` : Tensor parallel degree, it is recommended to just use num_gpus.

  `--batch-size BATCH_SIZE, -bs BATCH_SIZE` : Evalution batch size. Empirically, 2 works fastest.

  `--reward-bench` : Flag to run Reward Bench instead of the internal benchmarks.

  `--local-rank LOCAL_RANK` : For deepspeed, do not set this yourself.

  `--local_rank LOCAL_RANK` : For deepspeed, do not set this yourself.

### Showing Scores

To show a summary of the score run [score.py](./score.py). Pass in the evaluation output file prefix (should be the model name) into `--prefix`.  If the evaluation outputs are not in the root directory you are running `score.py` in, pass in the evaluation output directory to `--dir`.

#### Example

`python score.py --prefix Starling-RM-34B --dir ./evaluation_outputs`

## Some Errors

### Crashes Almost Immediately with Cryptic Cuda Errors
This is probably due to the code on one node not being identical to the host node.  Make sure the files are all the same or git is on the same branch and up to date.  Note, if your code is on a shared file system, this cannot be the issue.

### Crashes with Cryptic Map Error During Dataset Formatting
Run again on single node, cancel after dataset processing as completed.  Restart again; the error should clear.

### Training Hangs After Checkpoint Step
This has been observed with larger model training with no real remedy. The solution is just to restart at the checkpoint.

### NVCC Compatabilities Errors
If the NVCC version is to low or too high, Deepspeed will throw a tantrum.  To fix this, set the NVCC path variable to the correct version you downloaded or found laying around some where in the file system.  You should probably add this to your .bashrc file.

#### Oh No I Did What You Told Me to Do Above and It Still Breaks on Multi-Node Training
Sadly, Deepspeed is unable to piece together that running .bashrc might be important before starting up processes on child processes on different nodes.  Non NCCL environment variables will **NOT** carry over to the other node, nor will .bashrc run (before NVCC is called at least).  Therefore, any environment variables you need on both nodes will need to be set with `.deepspeed_env`.  To make the NVCC version carry over, assuming you have set NVCC in PATH, you can run something like this `echo PATH=$PATH >> .deepspeed_env` which will set up `.deepspeed_env` correctly.

More details can be found in the [Deepspeed Documentation](https://www.deepspeed.ai/getting-started/#multi-node-environment-variables).
