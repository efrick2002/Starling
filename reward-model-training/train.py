from os import environ
from os.path import abspath, dirname, join


ROOT_DIR = dirname(abspath(__file__))
PARENT_DIR = abspath(join(ROOT_DIR, ".."))
CACHE_DIR = join(PARENT_DIR, "cache")

environ["HF_DATASETS_CACHE"] = CACHE_DIR
environ["TRANSFORMERS_CACHE"] = CACHE_DIR
environ["PYTHONUNBUFFERED"] = "true"

import os
from transformers import Trainer, TrainingArguments
import random
import transformers
import argparse
import yaml
from models import create_reward_model_tokenizer, create_reward_model
from dataset import load_multiwise_dataset, DataCollatorReward, DatasetSplit
from utils import compute_metrics


def train(args):

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # required
    model_name = config["model_name"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    data_path = config["data_path"]
    max_length = config["max_length"]
    random_seed = config["seed"]
    deepspeed_config = config["deepspeed_config"]

    # optional
    data_percentage = config.get("data_percentage", 100)
    num_train_epochs = config.get("num_train_epochs", 1)
    lr_scheduler_type = config.get("lr_scheduler_type", "constant")
    warmup_ratio = config.get("warmup_ratio", 0.0)
    zero_sum_penalty = config.get("zero_sum_penalty", 0.0)
    use_cls = config.get("use_cls", False)
    chat_template = config.get("chat_template", None)

    if chat_template:
        with open(chat_template, 'r') as template_file:
            chat_template = template_file.readline()

    if args.checkpoint:
        resume_from_checkpoint = args.checkpoint
    else:
        resume_from_checkpoint = False

    random.seed(random_seed)
    transformers.set_seed(random_seed)

    output_path = (
        args.output_path + os.sep
        if not args.output_path.endswith(os.sep)
        else args.output_path
    )
    proj_name = f"reward-p{data_percentage}-s{str(random_seed)}-{model_name.split('/')[1]}{'-debug' if args.debug else ''}"
    output_dir = f"{output_path}rm-ckpt-{proj_name}"

    version = 1
    while os.path.exists(output_dir) and not resume_from_checkpoint:
        output_dir = output_dir.replace(f"_{version - 1}", "")
        output_dir = output_dir + f"_{version}"
        version += 1

    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",
        run_name=proj_name,
        num_train_epochs=num_train_epochs,
        logging_steps=1,
        gradient_accumulation_steps=(
            gradient_accumulation_steps if not args.debug else 2
        ),
        save_strategy="steps",
        evaluation_strategy="steps",
        ddp_timeout=9999999,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
        eval_steps=args.eval_steps if not args.debug else 1,
        save_steps=args.save_steps if not args.debug else 1,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_dir="./logs",
        fp16=False,
        bf16=True,
        learning_rate=learning_rate,
        deepspeed=deepspeed_config,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        do_train=True,
        bf16_full_eval=True,
        save_safetensors=True,
        save_on_each_node=args.save_on_each_node,
        local_rank=environ["LOCAL_RANK"],
    )

    print("TRAINING ARGS: ", training_args.to_dict())

    tokenizer = create_reward_model_tokenizer(model_name, use_cls)
    model = create_reward_model(
        model_name,
        tokenizer,
        use_cls=use_cls,
        k=7,
        zero_sum_penalty=zero_sum_penalty,
    ).from_pretrained(model_name)

    if args.debug:
        train_dataset = load_multiwise_dataset(
            data_path,
            DatasetSplit.DEBUG_TRAIN,
            tokenizer,
            args.no_cached_dataset,
            max_length=max_length,
            use_cls=use_cls,
            chat_template=chat_template,
        )
        val_dataset = load_multiwise_dataset(
            data_path,
            DatasetSplit.DEBUG_VALIDATION,
            tokenizer,
            args.no_cached_dataset,
            max_length=max_length,
            use_cls=use_cls,
            chat_template=chat_template,
        )
    else:
        if args.train_head_only:
            train_dataset = load_multiwise_dataset(
                data_path,
                DatasetSplit.PRETRAIN,
                tokenizer,
                args.no_cached_dataset,
                max_length=max_length,
                percentage=data_percentage,
                use_cls=use_cls,
                chat_template=chat_template,
            )
        else:
            train_dataset = load_multiwise_dataset(
                data_path,
                DatasetSplit.TRAIN,
                tokenizer,
                args.no_cached_dataset,
                max_length=max_length,
                percentage=data_percentage,
                use_cls=use_cls,
                chat_template=chat_template,
            )
        val_dataset = load_multiwise_dataset(
            data_path,
            DatasetSplit.VALIDATION,
            tokenizer,
            args.no_cached_dataset,
            max_length=max_length,
            use_cls=use_cls,
            chat_template=chat_template,
        )

    # Create the collator to gather batches of pairwise comparisons
    data_collator = DataCollatorReward()

    if args.train_head_only:
        print("Freezing Transformer Layers, Only Training Value Head.")
        model.freeze_transformer()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.deepspeed._config.zero_config.gather_16bit_weights_on_model_save = (
        True  # yeah, its that fking stupid
    )

    trainer.save_model()

    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # Read configuration from a YAML file

    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Config file containing training arguments.",
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Huggingface model name."
    )
    parser.add_argument(
        "--no-cached-dataset",
        "-ncd",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use this flag to avoid the cached dataset, helps avoid weird issues that might occur with caching.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Debug flag, runs through a train step, eval step, checkpoint save, train step, end model save.  Ignores save step, eval step, gradiant accumulation args and sets them to 1, 1, and 2, respectively.",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="For deepspeed, do not set this yourself.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="For deepspeed, do not set this yourself.",
    )
    parser.add_argument(
        "--checkpoint",
        "-cp",
        type=str,
        default=None,
        help="For restarting training at the checkpoint located at this inputted path.",
    )
    parser.add_argument(
        "--train-head-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use this flag to freeze the transformer block and train the value head only",
    )
    parser.add_argument(
        "--save-steps", type=int, default=10, help="How often to save a checkpoint."
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=10,
        help="How often to run an evaluation step.",
    )
    parser.add_argument(
        "--save-on-each-node",
        "-soen",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use when checkpoints are not being saved to a shared node.",
    )
    parser.add_argument(
        "--output-path",
        "-op",
        type=str,
        default="./",
        help="Use to specify a specific path in which to save the checkpoints (e.g. /scratch)",
    )
    parser.add_argument(
        "--save-total-limit",
        "-stl",
        type=int,
        default=2,
        help="The max amount of checkpoints to keep at any time.",
    )

    args = parser.parse_args()

    train(args)
