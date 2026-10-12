from dataclasses import fields

import torch


def collate_thd(samples, *, pad_token_id, alignment=1, attention_backend="auto"):
    from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
    from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs

    rows = []
    for sample in samples:
        tokens = sample["input_ids"]
        labels = torch.full_like(sample["labels"], -100)
        labels[:-1] = sample["labels"][1:]
        rows.append(
            dict(
                input_ids=tokens,
                labels=labels,
                loss_mask=(labels != -100).float(),
                position_ids=torch.arange(tokens.numel()),
            )
        )
    batch = build_mcore_thd_sequence_batch_from_rows(rows, pad_token_id=pad_token_id, pad_to_multiple_of=alignment)
    batch["sample_lengths"] = batch["cu_seqlens_q"].diff().long()
    if attention_backend == "flash" and "cu_seqlens_q_padded" in batch:
        # FA2 includes zero-loss alignment tokens in each sequence instead of ragged gaps.
        batch["cu_seqlens_q"] = batch.pop("cu_seqlens_q_padded")
        batch["cu_seqlens_kv"] = batch.pop("cu_seqlens_kv_padded")
    # The model computes per-sample MRoPE from visual grids and packed boundaries.
    batch["position_ids"] = None
    visual = {}
    for field in fields(GenericVisualInputs):
        values = [torch.as_tensor(sample[field.name]) for sample in samples if field.name in sample]
        if values:
            visual[field.name] = torch.cat(values, dim=0)
    batch["visual_inputs"] = GenericVisualInputs(**visual)
    return batch
