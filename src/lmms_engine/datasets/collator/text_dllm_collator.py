from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling

from ...utils import Logging


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get(
        "Asking-to-pad-a-fast-tokenizer", False
    )
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (
        pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
    ):
        if not isinstance(examples, torch.Tensor):
            return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer.pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


@dataclass
class TextDllmCollator(DataCollatorForLanguageModeling):
    def __init__(self, p_min, p_max, tokenizer, pad_to_multiple_of, return_tensors):
        super().__init__(
            mlm=True,
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.p_min = p_min
        self.p_max = p_max
        assert self.tokenizer.mask_token is not None
        assert 0.0 <= self.p_min <= self.p_max <= 1.0
        assert return_tensors == "pt", "This collator currently supports PyTorch only."

    def __call__(
        self, examples: List[Union[List[int], Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        if self.seed and self.generator is None:
            self.create_rng()

        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer,
                examples,
                return_tensors="pt",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        else:
            batch = {
                "input_ids": _torch_collate_batch(
                    examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
                )
            }
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch["input_ids"], batch["labels"], batch["mlm_prob"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        return batch

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens uniformly in each sequence for MLM training
        B = labels.shape[0]
        mlm_probabilities = (
            torch.rand(B, generator=self.generator, device=labels.device)
            * (self.p_max - self.p_min)
            + self.p_min
        )
        probability_matrix = mlm_probabilities.view(
            B, *([1] * (labels.ndim - 1))
        ) * torch.ones_like(labels, device=labels.device)

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(
            probability_matrix, generator=self.generator
        ).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )
        return inputs, labels, mlm_probabilities
