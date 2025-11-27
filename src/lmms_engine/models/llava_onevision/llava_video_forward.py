"""
LLaVA-Video Model Forward with Slow-Fast Frame Support

This module provides custom video processing methods for LlavaOnevisionModel that supports
slow-fast frame processing for video understanding.

Based on LLaVA-NeXT implementation and transformers v4.57.1 LlavaOnevisionModel.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionModel,
    LlavaOnevisionModelOutputWithPast,
)
from transformers.processing_utils import Unpack


def apply_2d_pool(
    model: LlavaOnevisionModel,
    image_features: torch.Tensor,
    stride: int = 2,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Apply 2D spatial pooling to video features with configurable stride.

    This follows LLaVA-NeXT's get_2dPool implementation.

    Args:
        model: LlavaOnevisionModel instance
        image_features: Features of shape (num_frames, seq_len, dim)
        stride: Pooling stride (2 for slow frames, 4 for fast frames)
        mode: Pooling mode - "bilinear", "average", or "max"

    Returns:
        Pooled features of shape (num_frames, reduced_seq_len, dim)
    """
    height = width = model.config.vision_config.image_size // model.config.vision_config.patch_size
    num_frames, seq_len, dim = image_features.shape

    # Reshape to 2D spatial layout
    image_features = image_features.view(num_frames, height, width, -1)
    image_features = image_features.permute(0, 3, 1, 2).contiguous()  # (num_frames, dim, height, width)

    # Apply pooling
    if mode == "average":
        image_features = nn.functional.avg_pool2d(image_features, stride)
    elif mode == "max":
        image_features = nn.functional.max_pool2d(image_features, stride)
    elif mode == "bilinear":
        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_features = nn.functional.interpolate(image_features, size=scaled_shape, mode="bilinear")
    else:
        raise ValueError(f"Unsupported pooling mode: {mode}")

    # Reshape back
    image_features = image_features.permute(0, 2, 3, 1).contiguous()
    image_features = image_features.view(num_frames, -1, dim)
    return image_features


def get_video_features_with_slow_fast(
    model: LlavaOnevisionModel,
    pixel_values: torch.FloatTensor,
    vision_feature_layer: Union[int, list[int]],
    vision_feature_select_strategy: str,
    add_faster_video: bool = False,
    faster_token_stride: int = 10,
    mm_spatial_pool_stride: int = 2,
    mm_spatial_pool_mode: str = "bilinear",
):
    """
    Extract video features with optional slow-fast frame support.

    This follows LLaVA-NeXT's encode_multimodals implementation.
    Key points:
    1. Pooling happens AFTER projector
    2. Slow frames use stride=mm_spatial_pool_stride (e.g., 2)
    3. Fast frames use stride=mm_spatial_pool_stride*2 (e.g., 4)
    4. Each frame gets a faster_token appended
    5. All frames are concatenated (not stacked!)

    Args:
        model: LlavaOnevisionModel instance
        pixel_values: Video frames of shape (batch_size, num_frames, channels, height, width)
        vision_feature_layer: Which layer(s) to extract features from
        vision_feature_select_strategy: Feature selection strategy
        add_faster_video: Whether to use slow-fast frames
        faster_token_stride: Stride for slow frames (every Nth frame is slow)
        mm_spatial_pool_stride: Base stride for spatial pooling
        mm_spatial_pool_mode: Pooling mode - "bilinear", "average", or "max"

    Returns:
        video_features: Processed video features of shape (batch_size, total_tokens, dim)
    """
    batch_size, frames, channels, height, width = pixel_values.shape
    pixel_values = pixel_values.view(batch_size * frames, channels, height, width)

    # Extract vision features from vision tower
    video_features = model.vision_tower(pixel_values, output_hidden_states=True)

    # Select vision features
    if isinstance(vision_feature_layer, int):
        selected_video_feature = video_features.hidden_states[vision_feature_layer]
    else:
        hs_pool = [video_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
        selected_video_feature = torch.cat(hs_pool, dim=-1)

    if vision_feature_select_strategy == "default":
        selected_video_feature = selected_video_feature[:, 1:]

    # Apply multimodal projector (BEFORE pooling, following LLaVA-NeXT)
    video_features = model.multi_modal_projector(selected_video_feature)

    # Reshape to (batch_size, frames, seq_len, dim)
    seq_len = video_features.shape[1]
    dim = video_features.shape[2]
    video_features = video_features.view(batch_size, frames, seq_len, dim)

    # Apply slow-fast frame processing if enabled
    if add_faster_video and mm_spatial_pool_stride > 1:
        # Process each video in the batch
        processed_videos = []

        for video_idx in range(batch_size):
            video_frames = video_features[video_idx]  # (frames, seq_len, dim)

            # Apply pooling to all frames to get slow and fast features
            slow_features = apply_2d_pool(model, video_frames, stride=mm_spatial_pool_stride, mode=mm_spatial_pool_mode)
            fast_features = apply_2d_pool(
                model, video_frames, stride=mm_spatial_pool_stride * 2, mode=mm_spatial_pool_mode
            )

            # Get faster_token if available
            faster_token = None
            if hasattr(model, "faster_token"):
                faster_token = model.faster_token[None]  # (1, dim)

            # Concatenate slow/fast features with faster_token for each frame
            concat_slow_fast_token = []
            for frame_idx in range(frames):
                if frame_idx % faster_token_stride == 0:
                    # Slow frame: use high-resolution features
                    frame_feat = slow_features[frame_idx]  # (seq_len_slow, dim)
                else:
                    # Fast frame: use low-resolution features
                    frame_feat = fast_features[frame_idx]  # (seq_len_fast, dim)

                # Add faster_token to end of frame if available
                if faster_token is not None:
                    frame_feat = torch.cat(
                        (frame_feat, faster_token.to(frame_feat.device)), dim=0
                    )  # (seq_len + 1, dim)

                concat_slow_fast_token.append(frame_feat)

            # Concatenate all frames sequentially (following LLaVA-NeXT line 326)
            video_feat = torch.cat(concat_slow_fast_token, dim=0)  # (total_tokens, dim)
            processed_videos.append(video_feat)

        # Find max length for padding (if needed in batch)
        max_len = max(v.shape[0] for v in processed_videos)

        # Pad videos to same length if necessary
        if len(set(v.shape[0] for v in processed_videos)) > 1:
            padded_videos = []
            for video_feat in processed_videos:
                if video_feat.shape[0] < max_len:
                    padding = torch.zeros(
                        max_len - video_feat.shape[0], dim, device=video_feat.device, dtype=video_feat.dtype
                    )
                    video_feat = torch.cat((video_feat, padding), dim=0)
                padded_videos.append(video_feat)
            video_features = torch.stack(padded_videos, dim=0)  # (batch_size, max_len, dim)
        else:
            video_features = torch.stack(processed_videos, dim=0)  # (batch_size, total_tokens, dim)

        return video_features
    else:
        # Standard pooling without slow-fast
        processed_videos = []
        for video_idx in range(batch_size):
            video_frames = video_features[video_idx]  # (frames, seq_len, dim)
            pooled_frames = apply_2d_pool(model, video_frames, stride=mm_spatial_pool_stride, mode=mm_spatial_pool_mode)
            # Concatenate all frames
            video_feat = pooled_frames.reshape(-1, dim)  # (frames * reduced_seq_len, dim)
            processed_videos.append(video_feat)

        video_features = torch.stack(processed_videos, dim=0)  # (batch_size, frames * reduced_seq_len, dim)
        return video_features


def forward(
    self: LlavaOnevisionModel,
    input_ids: Optional[torch.LongTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[torch.LongTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_sizes_videos: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    vision_feature_layer: Optional[Union[int, list[int]]] = None,
    vision_feature_select_strategy: Optional[str] = None,
    vision_aspect_ratio: Optional[str] = None,
    batch_num_images: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Union[tuple, LlavaOnevisionModelOutputWithPast]:
    """
    Forward pass with slow-fast frame support for LlavaOnevisionModel.

    This extends the standard forward to support:
    - Slow frames: High-resolution features (stride=2)
    - Fast frames: Low-resolution features (stride=4)
    - faster_token: Learnable token to mark frame type
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    vision_feature_layer = (
        vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
    )
    vision_feature_select_strategy = (
        vision_feature_select_strategy
        if vision_feature_select_strategy is not None
        else self.config.vision_feature_select_strategy
    )
    vision_aspect_ratio = vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio

    # Get slow-fast frame configuration
    faster_token_stride = getattr(self.config, "faster_token_stride", 10)
    mm_spatial_pool_stride = getattr(self.config, "mm_spatial_pool_stride", 2)
    mm_spatial_pool_mode = getattr(self.config, "mm_spatial_pool_mode", "bilinear")

    # Check if slow-fast is enabled by checking if faster_token exists
    add_faster_video = hasattr(self, "faster_token")

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    # Process images (standard processing, no slow-fast)
    if pixel_values is not None:
        image_features = self.get_image_features(
            pixel_values,
            image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            batch_num_images=batch_num_images,
        )
        image_features = torch.cat(image_features, dim=0)
        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
        special_image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

    # Process videos with slow-fast frame support
    if pixel_values_videos is not None:
        video_features = get_video_features_with_slow_fast(
            self,
            pixel_values_videos,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            add_faster_video=add_faster_video,
            faster_token_stride=faster_token_stride,
            mm_spatial_pool_stride=mm_spatial_pool_stride,
            mm_spatial_pool_mode=mm_spatial_pool_mode,
        )

        # Add image newline tokens
        image_newline = (
            self.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(video_features.device)
        )
        video_features = torch.cat((video_features, image_newline), dim=1)

        video_features = video_features.flatten(0, 1).to(inputs_embeds.device, inputs_embeds.dtype)
        _, special_video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_features
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

    # Forward through language model
    outputs = self.language_model(
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    return LlavaOnevisionModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=image_features if pixel_values is not None else None,
        video_hidden_states=video_features if pixel_values_videos is not None else None,
    )
