from os import environ
from os.path import abspath, dirname, join

ROOT_DIR = dirname(abspath(__file__))
PARENT_DIR = abspath(join(ROOT_DIR, ".."))
CACHE_DIR = join(PARENT_DIR, "cache")

environ["HF_DATASETS_CACHE"] = CACHE_DIR
environ["TRANSFORMERS_CACHE"] = CACHE_DIR
environ["PYTHONUNBUFFERED"] = "true"

from datasets import load_dataset, Dataset as HF_Dataset
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from enum import Enum
from transformers import PreTrainedTokenizerBase
import re
from typing import Callable


class DatasetSplit(Enum):
    PRETRAIN = "pretrain"
    TRAIN = "train"
    VALIDATION = "val"
    DEBUG_TRAIN = "debug_train"
    DEBUG_VALIDATION = "debug_val"


class ChatRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def _get_split(path: str, split: str, percentage: int) -> HF_Dataset:
    """Returns dataset for the given split.  Implements the train/validation split logic.

    Args:
        path (str): HF model path.
        split (str): Dataset split (see enum for values)
        percentage (int): Percentage of data to train on.
        dataset_cache_dir (str, optional): Whether to use HF dataset cache. Defaults to None.

    Returns:
        HF_Dataset: HF dataset of the correct split.
    """
    match split:
        case DatasetSplit.TRAIN:
            dataset = load_dataset(
                path,
                split=f"train[:{'-2000' if percentage == 100 else str(percentage) + '%'}]",
            )
        case DatasetSplit.PRETRAIN:
            dataset = load_dataset(path, split="train[-2000:-1000]")

        case DatasetSplit.VALIDATION:
            dataset = load_dataset(path, split="train[-1000:]")

        case DatasetSplit.DEBUG_TRAIN:
            dataset = load_dataset(path, split="train[:32]")

        case DatasetSplit.DEBUG_VALIDATION:
            dataset = load_dataset(path, split="train[-32:]")

    return dataset


def _nectar_to_message_format(prompt: str) -> list:
    """Converts the Nectar dataset to messages format.

    Args:
        prompt (str): Input prompt, can be multi-turn with format "\\n\\nHuman: ... \\n\\nAssistant: ".

    Returns:
        list: message format, [{'role': <role>, 'content': <content>} ... ]
    """
    split_prompt = re.split(r"\n\nHuman: |\n\nAssistant: ", prompt)[1:-1]

    messages = []

    for i, utterance in enumerate(split_prompt):
        messages.append(
            {
                "role": ChatRole.USER.value if i % 2 == 0 else ChatRole.ASSISTANT.value,
                "content": utterance,
            }
        )

    return messages


def _create_formatter(
    tokenizer: PreTrainedTokenizerBase, use_cls=False, chat_template=None
) -> Callable:
    """Creates the Nectar dataset formatter which takes in a row and returns the formatted text input for training.

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer for the model.
        use_cls (bool, optional): Set True to append <|cls|>. Defaults to False.
        chat_template (str, optional): Chat template for formatting, leave as None to use default template.  Check the default template before using it! Defaults to None.

    Returns:
        Callable: Formatter that only takes in a row and returns the formatted text.
    """

    def formatter(sample):

        messages = _nectar_to_message_format(sample["prompt"])

        output = []

        for answer in sample["answers"]:

            messages_with_answer = messages + [
                {"role": ChatRole.ASSISTANT.value, "content": answer["answer"]}
            ]

            formatted_answer = tokenizer.apply_chat_template(
                messages_with_answer, tokenize=False, chat_template=chat_template
            )

            output.append(formatted_answer)

        # Append CLS token here.
        if use_cls:
            output = [text + "<|cls|>" for text in output]

        return {"formatted_answers": output}

    return formatter


def load_multiwise_dataset(
    path: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    no_cache: bool,
    max_length=2000,
    percentage=100,
    use_cls=False,
    chat_template=None,
) -> Dataset:
    """Loads the dataset for training. Handles split, formatting, and tokenizing.

    Args:
        path (str): HF model path.
        split (str): Dataset split (see DatasetSplit enum).
        tokenizer (PreTrainedTokenizerBase): Model tokenizer.
        no_cache (bool): Set False to avoid using HF dataset cache.
        max_length (int, optional): Maximum token length of each training point. Defaults to 2000.
        percentage (int, optional): Training data percentage. Defaults to 100.
        use_cls (bool, optional): Set True to append <|cls|>. Defaults to False.
        chat_template (str, optional): Chat template for formatting, leave as None to use default template.  Check the default template before using it! Defaults to None.

    Returns:
        Dataset: The final formatted torch dataset, containing tokenizer inputs.
    """

    use_cache = not no_cache
    data = _get_split(path, split, percentage)
    formatted = data.map(
        _create_formatter(tokenizer, use_cls=use_cls, chat_template=chat_template),
        num_proc=32,
        load_from_cache_file=use_cache,
        desc=f"Formatting {split} split to format.",
    )
    print("Example formatted inputs: ", formatted[3]["formatted_answers"])

    tokenized = formatted.map(
        lambda p: tokenizer(
            p,
            truncation=True,
            max_length=max_length,
            padding="longest",
            return_tensors="pt",
        ),
        input_columns="formatted_answers",
        num_proc=32,
        load_from_cache_file=use_cache,
        desc=f"Tokenizing {split} split",
    ).remove_columns(
        ["prompt", "answers", "turns", "num_responses", "source", "formatted_answers"]
    )

    return tokenized.with_format("torch")


class DataCollatorReward:
    def __call__(self, data):
        batch = {}
        batch["input_ids"] = torch.vstack(
            [item for f in data for item in f["input_ids"]]
        )
        batch["attention_mask"] = torch.vstack(
            [item for f in data for item in f["attention_mask"]]
        )

        # TODO: Make this automatically pick the right k-wise.
        batch["labels"] = torch.ones((len(data), 7, 7))
        return batch
