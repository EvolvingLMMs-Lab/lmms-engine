from typing import Dict

import os
import torch


def validate_bagel_batch(batch: Dict[str, torch.Tensor]) -> None:
    """Lightweight runtime checks for Bagel batch integrity.

    Raises AssertionError on inconsistencies. Intended for debugging; guard
    any calls to this with an env or config flag to avoid overhead.
    """
    # Sequence length and sample lens
    seq_len = int(batch["sequence_length"]) if torch.is_tensor(batch["sequence_length"]) else int(batch["sequence_length"])
    sample_lens = batch.get("sample_lens", None)
    assert sample_lens is not None, "missing sample_lens"
    assert seq_len == int(sample_lens.sum().item()), "sequence_length must equal sum(sample_lens)"

    # Text packing pieces
    for k in ["packed_text_ids", "packed_text_indexes", "packed_position_ids"]:
        assert k in batch, f"missing {k}"
    assert batch["packed_text_ids"].shape[0] == batch["packed_text_indexes"].shape[0], "packed_text_ids/indexes length mismatch"

    # Index bounds
    max_index = 0
    for k in ["packed_text_indexes", "packed_vae_token_indexes", "packed_vit_token_indexes"]:
        if k in batch:
            assert batch[k].ndim == 1, f"{k} must be 1D"
            max_index = max(max_index, int(batch[k].max().item()) if batch[k].numel() > 0 else 0)
    assert max_index < seq_len, "some packed indexes exceed sequence_length"

    # VAE tokens shape alignment
    if "packed_vae_token_indexes" in batch:
        # shapes list of tuples
        shapes = batch.get("patchified_vae_latent_shapes", [])
        if isinstance(shapes, torch.Tensor):
            shapes = [tuple(map(int, s.tolist())) for s in shapes]
        total_latent = sum(h * w for (h, w) in shapes)
        assert total_latent == batch["packed_vae_token_indexes"].shape[0], "VAE token indexes length mismatch"
        for k in ["packed_timesteps", "mse_loss_indexes"]:
            assert k in batch, f"missing {k} for VAE generation"
            assert batch[k].shape[0] == total_latent, f"{k} length mismatch"

    # ViT tokens shape alignment
    if "packed_vit_token_indexes" in batch:
        for k in ["packed_vit_tokens", "packed_vit_position_ids", "vit_token_seqlens"]:
            assert k in batch, f"missing {k} for ViT path"
        n = batch["packed_vit_token_indexes"].shape[0]
        assert batch["packed_vit_tokens"].shape[0] == n, "packed_vit_tokens length mismatch"
        assert batch["packed_vit_position_ids"].shape[0] == n, "packed_vit_position_ids length mismatch"

    # Per-sample masks
    masks = batch.get("nested_attention_masks", None)
    assert masks is not None, "missing nested_attention_masks"
    assert len(masks) == sample_lens.shape[0], "mask list length must equal batch size"
    for i, (m, sl) in enumerate(zip(masks, sample_lens.tolist())):
        assert m.shape[0] == m.shape[1] == sl, f"mask[{i}] shape must be ({sl},{sl})"


def validate_bagel_batch_if_enabled(batch: Dict[str, torch.Tensor]) -> None:
    if os.environ.get("LMMS_BAGEL_VALIDATE", "0") == "1":
        validate_bagel_batch(batch)

