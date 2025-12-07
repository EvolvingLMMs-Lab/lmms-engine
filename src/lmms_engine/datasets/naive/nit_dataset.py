import time
from collections import defaultdict

import numpy as np
from datasets import load_dataset

from lmms_engine.datasets.config import DatasetConfig
from lmms_engine.mapping_func import register_dataset

from ..processor.nit_processor import NitProcessor

#############################################
#                   LPFHP                   #
#############################################

# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# modified from https://github.com/graphcore/examples/blob/v3.2.0/tutorials/blogs_code/packedBERT/lpfhp.py
"""Longest-pack-first histogram-packing."""


def add_pack(pack, count, tmp, final, limit, offset, max_sequence_length=512):
    """Filter out packs that reached maximum length or number of components."""
    # sanity checks
    assert max_sequence_length - sum(pack) == offset, "Incorrect offset."
    assert offset >= 0, "Too small offset."
    assert offset < max_sequence_length, "Too large offset."
    if len(pack) == limit or offset == 0:
        final[offset].append((count, pack))
    else:
        tmp[offset].append((count, pack))


def LPFHP(histogram, max_sequence_length, max_sequences_per_pack, distribute=True):
    """Longest-pack-first histogram-packing."""
    start = time.time()
    reversed_histogram = np.flip(histogram)
    # Initialize main strategy data dictionary.
    # The key indicates how many tokens are left for full length.
    # The value is a list of tuples, consisting of counts and respective packs.
    # A pack is a (sorted) list of sequence length values that get concatenated.
    tmp_strategies_per_length = defaultdict(list)
    strategies_per_length = defaultdict(list)
    if max_sequences_per_pack == "max":
        max_sequences_per_pack = max_sequence_length
    # Index i indicates here, how much space is left, due to reversed histogram
    for i in range(max_sequence_length):
        n_sequences_to_bin = reversed_histogram[i]
        length_to_bin = max_sequence_length - i
        offset = 0  # smallest possible offset for perfect fit
        while n_sequences_to_bin > 0:
            if (length_to_bin + offset) in tmp_strategies_per_length:
                # extract worst pack that will get modified
                n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()
                # calculate how often the current sequence maximally fits in
                repeat = min(1 + offset // length_to_bin, max_sequences_per_pack - len(pack))
                # correct dependent on count
                while n_sequences_to_bin // repeat == 0:
                    repeat -= 1
                if not distribute:
                    repeat = 1
                new_pack = pack + [length_to_bin] * repeat
                count = min(n_sequences_to_pack, n_sequences_to_bin // repeat)
                if n_sequences_to_pack > count:
                    # old pack gets reduced
                    n_sequences_to_pack -= count
                    tmp_strategies_per_length[length_to_bin + offset].append((n_sequences_to_pack, pack))
                    n_sequences_to_bin -= count * repeat
                else:
                    n_sequences_to_bin -= n_sequences_to_pack * repeat
                add_pack(
                    new_pack,
                    count,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    offset - (repeat - 1) * length_to_bin,
                    max_sequence_length,
                )
                # clean up to speed up main key search
                if not tmp_strategies_per_length[length_to_bin + offset]:
                    tmp_strategies_per_length.pop(length_to_bin + offset)
                # reset offset in case best fit changed
                offset = 0
            else:
                offset += 1
            # Does not fit anywhere. Create new pack.
            if offset >= max_sequence_length - length_to_bin + 1:
                # similar repetition but no dependence on pack.
                repeat = min(max_sequence_length // length_to_bin, max_sequences_per_pack)
                while n_sequences_to_bin // repeat == 0:
                    repeat -= 1
                if not distribute:
                    repeat = 1
                add_pack(
                    [length_to_bin] * repeat,
                    n_sequences_to_bin // repeat,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    max_sequence_length - length_to_bin * repeat,
                    max_sequence_length,
                )
                n_sequences_to_bin -= n_sequences_to_bin // repeat * repeat
    # merge all strategies
    for key in tmp_strategies_per_length:
        strategies_per_length[key].extend(tmp_strategies_per_length[key])
    # flatten strategies dictionary
    strategy_set = []
    strategy_repeat_count = []
    for key in strategies_per_length:
        for count, pack in strategies_per_length[key]:
            pack.reverse()
            strategy_set.append(pack)
            strategy_repeat_count.append(count)

    # Summarize efficiency of solution
    duration = time.time() - start
    sequence_lengths = np.arange(1, max_sequence_length + 1)
    strategy_repeat_count = np.array(strategy_repeat_count)
    n_strategies = len(strategy_set)
    old_number_of_samples = histogram.sum()
    new_number_of_samples = strategy_repeat_count.sum()
    sequences = sum([count * len(pack) for count, pack in zip(strategy_repeat_count, strategy_set)])
    total_tokens = max_sequence_length * new_number_of_samples
    empty_tokens = sum(
        [count * (max_sequence_length - sum(pack)) for count, pack in zip(strategy_repeat_count, strategy_set)]
    )
    efficiency = 100 - empty_tokens / total_tokens * 100
    speedup_upper_bound = 1.0 / (
        1 - (histogram * (1 - sequence_lengths / max_sequence_length)).sum() / old_number_of_samples
    )

    print(
        f"Packing efficiency (fraction of real tokens): {efficiency:3.4f}\n",
        f"Speed-up theoretical limit: {speedup_upper_bound:3.4f}\n",
        f"Achieved speed-up over un-packed dataset: {old_number_of_samples / new_number_of_samples:3.5f}",
        f"Runtime: Packed {old_number_of_samples} sequences in {duration:3.3f} seconds.",
    )

    return strategy_set, strategy_repeat_count


#############################################
#                   NitDataset              #
#############################################


@register_dataset("nit")
class NitDataset:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.processor = NitProcessor(config)

    def _build_from_config(self):
        # A bit ugly, but it seems that I cannot merge with multimodal dataset
        if self.config.dataset_format == "hf_dataset":
            self.dataset = load_dataset(self.config.dataset_path, split="train")
        else:
            raise NotImplementedError("Only hf_dataset is supported for now")

        self.dataset = self.dataset.map(self.processor.process, num_proc=self.config.processor_workers)
        self.data_lens = self.dataset["num_tokens"]

        if self.config.packing:
            histogram = np.zeros(self.config.packing_length + 1, dtype=int)
            for length in self.data_lens:
                if length <= self.config.packing_length:
                    histogram[length] += 1

            max_sequences_per_pack = getattr(self.config, "max_sequences_per_pack", "max")
            strategy_set, strategy_repeat_count = LPFHP(
                histogram,
                self.config.packing_length,
                max_sequences_per_pack=max_sequences_per_pack,
                distribute=True,
            )

            indices_by_length = defaultdict(list)
            for idx, length in enumerate(self.data_lens):
                if length <= self.config.packing_length:
                    indices_by_length[length].append(idx)

            self.packed_indices = []
            for count, pack in zip(strategy_repeat_count, strategy_set):
                for _ in range(count):
                    current_pack_indices = []
                    for length in pack:
                        if indices_by_length[length]:
                            current_pack_indices.append(indices_by_length[length].pop())
                        else:
                            raise ValueError(f"Not enough sequences of length {length} for packing.")
                    self.packed_indices.append(current_pack_indices)

    def __getitem__(self, index):
        if self.config.packing:
            indices = self.packed_indices[index]
            return [self.dataset[idx] for idx in indices]
        return self.dataset[index]

    def __len__(self):
        if self.config.packing:
            return len(self.packed_indices)
        return len(self.dataset)
