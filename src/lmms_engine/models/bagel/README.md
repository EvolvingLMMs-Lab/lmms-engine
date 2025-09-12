Bagel Model (Qwen2 + SigLIP + VAE)

Overview
- Visual understanding: user images are patchified via SigLIP (ViT) and fused into the LLM sequence.
- Visual generation: assistant images are tokenized by a convolutional VAE and generated via flow steps.
- Text: standard chat segments using Qwen2 tokenizer with special tokens: <|im_start|>, <|im_end|>, <|vision_start|>, <|vision_end|>.

Processor Outputs (per sample)
- sequence_length: int, total tokens in the packed sequence.
- sample_lens: [int], length per sample; in batch this is a tensor of size (B,).
- packed_text_ids: [T_text] LongTensor.
- packed_text_indexes: [T_text] LongTensor, positions of text tokens in packed sequence [0..sequence_length-1].
- packed_position_ids: [sequence_length] LongTensor, RoPE positions per token.
- nested_attention_masks: list of FloatTensor[L_i, L_i], one per sample; values are 0.0 for attend, -inf for mask.

Visual Understanding (ViT) — present when user images exist
- packed_vit_tokens: [N_vit, D] FloatTensor, patchified ViT tokens.
- packed_vit_position_ids: [N_vit] LongTensor, 2D positions flattened to 1D.
- packed_vit_token_indexes: [N_vit] LongTensor, indexes into packed sequence.
- vit_token_seqlens: [B_img] IntTensor, token counts per image.

Visual Generation (VAE) — present when assistant images exist
- padded_images: FloatTensor[B_img, C, H_max, W_max], padded original images before VAE encode.
- patchified_vae_latent_shapes: list[(h, w)] per image, latent grid sizes.
- packed_vae_token_indexes: [N_vae] LongTensor, indexes into packed sequence.
- packed_latent_position_ids: [N_vae] LongTensor, 2D latent positions flattened to 1D.
- packed_timesteps: [N_vae] FloatTensor, flow timesteps per latent token.
- mse_loss_indexes: [N_vae] Bool/LongTensor, mask of positions to apply MSE.

Conventions
- Causal attention for text segments; full attention for ViT tokens; noise-block for VAE latents.
- Assistant text is labeled for CE targets; user/system text is masked.
- Position ids: text uses sequential RoPE; images (both ViT and VAE) use a single global position per image token group.

Validation
- Set `LMMS_BAGEL_VALIDATE=1` to enable runtime shape/index checks in the collator.

Performance
- Set `LMMS_BAGEL_COMPILE_FLEX_ATTENTION=1` or pass `compile_flex_attention: true` in `training_config` to compile PyTorch Flex Attention for speed.

