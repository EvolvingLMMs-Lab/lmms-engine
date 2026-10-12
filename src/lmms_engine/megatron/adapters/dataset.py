import math
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
from megatron.bridge.data.base import DatasetBuildContext, DatasetProvider

from .batch import collate_thd


class PackedIterator:
    def __init__(
        self, dataset, *, rank, world_size, seq_length, alignment, packing, seed, shuffle, attention_backend="auto"
    ):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seq_length = seq_length
        self.alignment = alignment
        self.packing = packing
        self.seed = seed
        self.shuffle = shuffle
        self.attention_backend = attention_backend
        self.total_tokens = 0
        self.cursor = rank
        self.epoch = -1
        self.order = None
        if len(dataset) < world_size:
            raise ValueError("The dataset must contain at least one sample per data-parallel rank")

    def __iter__(self):
        return self

    def _sample(self):
        epoch, offset = divmod(self.cursor, len(self.dataset))
        if epoch != self.epoch:
            self.epoch = epoch
            generator = torch.Generator().manual_seed(self.seed + epoch)
            self.order = torch.randperm(len(self.dataset), generator=generator) if self.shuffle else None
        index = int(self.order[offset]) if self.order is not None else offset
        self.cursor += self.world_size
        return self.dataset[index]

    def __next__(self):
        samples, used = [], 0
        scanned = 0
        while True:
            cursor = self.cursor
            sample = self._sample()
            length = math.ceil(sample["input_ids"].numel() / self.alignment) * self.alignment
            if length > self.seq_length:
                scanned += 1
                if scanned >= len(self.dataset):
                    raise ValueError("No sample fits seq_length; increase it or lower image_max_pixels")
                continue
            scanned = 0
            if samples and used + length > self.seq_length:
                self.cursor = cursor
                break
            if not torch.any(sample["labels"][1:] != -100):
                raise ValueError("A training sample has no supervised next-token targets")
            samples.append(sample)
            used += length
            if not self.packing or used == self.seq_length:
                break
        batch = collate_thd(
            samples,
            pad_token_id=self.dataset.processor.tokenizer.pad_token_id,
            alignment=self.alignment,
            attention_backend=self.attention_backend,
        )
        self.total_tokens += int(batch["sample_lengths"].sum())
        batch["total_consumed_tokens"] = self.total_tokens
        return batch

    def save_state(self):
        return dict(
            cursor=self.cursor,
            total_tokens=self.total_tokens,
            rank=self.rank,
            world_size=self.world_size,
            seed=self.seed,
            size=len(self.dataset),
            seq_length=self.seq_length,
            alignment=self.alignment,
            packing=self.packing,
            shuffle=self.shuffle,
            attention_backend=self.attention_backend,
        )

    def restore_state(self, state):
        expected = self.save_state()
        for key in expected.keys() - {"cursor", "total_tokens"}:
            if state.get(key) != expected[key]:
                raise ValueError(f"Cannot restore data iterator with changed {key}")
        self.cursor = state["cursor"]
        self.total_tokens = state.get("total_tokens", 0)
        self.epoch = -1


@dataclass
class LMMSDatasetProvider(DatasetProvider):
    lmms_dataset: dict = field(default_factory=dict)
    seq_length: int = 4096
    alignment: int = 1
    online_packing: bool = True
    attention_backend: str = "auto"
    dataloader_type: str = "external"
    num_workers: int = 0

    def build_datasets(self, context: DatasetBuildContext):
        from lmms_engine.datasets import DatasetConfig
        from lmms_engine.datasets.naive.qwen3_vl_dataset import Qwen3VLDataset

        config = DatasetConfig(**self.lmms_dataset)
        shuffle, seed = config.shuffle, config.data_seed or 0
        config.packing = False
        config.shuffle = False
        config.filter_overlong = False
        dataset = Qwen3VLDataset(config)
        dataset.build()
        group = context.pg_collection.dp
        rank, world_size = dist.get_rank(group), dist.get_world_size(group)
        iterator = PackedIterator(
            dataset,
            rank=rank,
            world_size=world_size,
            seq_length=self.seq_length,
            alignment=self.alignment,
            packing=self.online_packing,
            seed=seed,
            shuffle=shuffle,
            attention_backend=self.attention_backend,
        )
        return iterator, None, None
