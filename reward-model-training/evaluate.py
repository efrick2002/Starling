from os import environ
from os.path import abspath, dirname, join


ROOT_DIR = dirname(abspath(__file__))
PARENT_DIR = abspath(join(ROOT_DIR, ".."))
CACHE_DIR = join(PARENT_DIR, "cache")

environ["HF_DATASETS_CACHE"] = CACHE_DIR
environ["TRANSFORMERS_CACHE"] = CACHE_DIR
environ["PYTHONUNBUFFERED"] = "true"

import argparse
from models import create_reward_model_tokenizer, create_reward_model
import deepspeed
from transformers import pipeline, TextClassificationPipeline
import time
from typing import Dict
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch
from benchmark_dataset import INTERAL_BENCHMARK_REGISTRY, RewardBench


def main(args):

    LOCAL_RANK = int(environ["LOCAL_RANK"])

    if args.chat_template:
        with open(args.chat_template, "r") as template_file:
            chat_template = template_file.readline()
    else:
        chat_template = None

    class RewardPipeline(TextClassificationPipeline):
        def preprocess(self, inputs, **tokenizer_kwargs) -> Dict[str, torch.Tensor]:
            return_tensors = self.framework

            formatted = self.tokenizer.apply_chat_template(
                inputs, tokenize=False, chat_template=chat_template
            )

            if args.use_cls:
                formatted = formatted + "<|cls|>"

            return self.tokenizer(
                formatted,
                return_tensors=return_tensors,
                max_length=2000,
                padding="longest",
                **tokenizer_kwargs,
            )

        def postprocess(
            self, model_outputs, function_to_apply=None, top_k=1, _legacy=True
        ):
            return model_outputs["scores"].float().cpu()

    reward_tokenizer = create_reward_model_tokenizer(
        args.base_model, use_cls=args.use_cls
    )

    model = create_reward_model(
        args.base_model,
        tokenizer=reward_tokenizer,
        use_cls=args.use_cls,
        inference=True,
    ).from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    pipe = pipeline(
        task="text-classification",
        model=model,
        tokenizer=reward_tokenizer,
        pipeline_class=RewardPipeline,
        batch_size=args.batch_size,
    )

    pipe.device = torch.device(f"cuda:{LOCAL_RANK}")

    pipe.model = deepspeed.init_inference(
        pipe.model,
        tensor_parallel={"tp_size": args.tensor_parallel_size},
        dtype=torch.bfloat16,
        checkpoint=None,
        quant={"enabled": False},
    )

    if args.reward_bench:
        benchmarks = [RewardBench]
    else:
        benchmarks = INTERAL_BENCHMARK_REGISTRY

    for benchmark in benchmarks:
        dataset = benchmark()

        tick = time.time()
        with torch.no_grad():
            output = [
                out
                for out in tqdm(
                    pipe(dataset, batch_size=args.batch_size), total=len(dataset)
                )
            ]

        if LOCAL_RANK == 0:
            tock = time.time()

            print("TIME TAKEN: ", tock - tick)

            rewards = torch.concat(output).numpy().tolist()

            print("First 10 Rewards: ", rewards[:10])

            dataset.df["score_a" if not args.reward_bench else "score_chosen"] = (
                rewards[:: dataset.responses_per_question]
            )
            dataset.df["score_b" if not args.reward_bench else "score_rejected"] = (
                rewards[1 :: dataset.responses_per_question]
            )

            if dataset.responses_per_question == 3:
                dataset.df["score_c"] = rewards[2 :: dataset.responses_per_question]

            name = (
                f"{args.model_path.split('/')[-1]}"
                if "/" in args.model_path
                else args.model_path
            )

            dataset.df.to_json(
                f"{name}_{dataset.benchmark_name}.json", orient="records", indent=1
            )

            if args.reward_bench:
                dataset.df["correct"] = (
                    dataset.df["score_chosen"] > dataset.df["score_rejected"]
                )

                score_table = dataset.df.groupby("subset").agg({"correct": "mean"})

                print(score_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Huggingface model path."
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base model used to train the model, needs to be registered.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="Path to the chat template to use, if None, the default for the tokenizer is used.",
    )
    parser.add_argument(
        "--use-cls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag for adding the cls token to the tokenizer and end of the each query.",
    )
    parser.add_argument(
        "--reward-bench",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag to run Reward Bench instead of the internal benchmarks.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        "-tp",
        type=int,
        default=4,
        help="Tensor parallel degree, it is recommended to just use num_gpus.",
    )
    parser.add_argument(
        "--batch-size",
        "-bs",
        type=int,
        default=2,
        help="Evalution batch size.  Empirically, 2 works fastest.",
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

    args = parser.parse_args()

    main(args)
