import json
import os
import time
from functools import partial
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from trlx.data.p3o_types import P3ORLBatch, P3ORLElement
from trlx.pipeline import BaseRolloutStore


class P3ORolloutStorage(BaseRolloutStore):
    """
    Rollout storage for training P3O
    """

    def __init__(self, pad_token_id, padding_side):
        super().__init__()

        self.pad_token_id = pad_token_id
        self.padding_side = padding_side
        self.history: Iterable[P3ORLElement] = []

    def push(self, exps: Iterable[P3ORLElement]):
        self.history += exps

    def clear_history(self):
        self.history = []

    def export_history(self, location: str, only_text=True):
        assert os.path.exists(location)

        fpath = os.path.join(location, f"epoch-{str(time.time())}.json")

        def exp_to_dict(exp):
            return {k: v.cpu().tolist() for k, v in exp.__dict__.items()}

        def filter_text(d, only_text):
            if only_text:
                keys = list(d.keys())
                for key in keys:
                    if key != "query_tensor" and key != "response_tensor":
                        d.pop(key)
            return d

        data = [filter_text(exp_to_dict(exp), only_text) for exp in self.history]
        with open(fpath, "w") as f:
            f.write(json.dumps(data, indent=2))

    def __getitem__(self, index: int) -> P3ORLElement:
        return self.history[index]

    def __len__(self) -> int:
        return len(self.history)

    def create_loader(
        self,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        def collate_fn(elems: Iterable[P3ORLElement]):
            if self.padding_side == "right":
                # Right padding of already right-padded queries
                query_tensors = pad_sequence(
                    [elem.query_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                )
            else:
                # Left padding of already left-padded queries
                query_tensors = pad_sequence(
                    [elem.query_tensor.flip(0) for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ).flip(1)

            num_samples = len(elems)
            pad_response_tensors = pad_sequence(
                [elem.response_tensor[0] for elem in elems] + [elem.response_tensor[1] for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            )
            pad_response_tensors = torch.stack([pad_response_tensors[:num_samples], pad_response_tensors[num_samples:]])

            return P3ORLBatch(
                query_tensors,
                pad_response_tensors,
                torch.stack([elem.logratios for elem in elems]).transpose(0, 1),
                torch.stack([elem.scalar_rewards for elem in elems]).transpose(0, 1),
            )

        return DataLoader(self, batch_size, shuffle=shuffle, collate_fn=collate_fn)


elem = P3ORLElement(
    query_tensor=torch.zeros([78]),
    response_tensor=torch.zeros([2, 100]),
    logratios=torch.zeros([2]),
    scalar_rewards=torch.zeros([2]),
)
print(elem.query_tensor.shape)
datastore = P3ORolloutStorage(pad_token_id=32000, padding_side="right")
datastore.push([elem])
datastore.push([elem])
datastore.push([elem])
datastore.push([elem])
print(len(datastore.history))

dataloader = datastore.create_loader(batch_size=1, shuffle=False)
for batch in dataloader:
    print(batch.response_tensors.shape)
