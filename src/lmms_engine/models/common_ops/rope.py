from typing import Optional

import numpy as np
import torch
from transformers import Qwen2_5_VLModel, Qwen3VLModel


def qwen3_vl_get_rope_index(
    self: Qwen3VLModel,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids.

    Performance notes: the trivial Python-loop port of the upstream layout
    triggers O(B + N_vision) device syncs per call (input_ids.tolist(),
    repeat_interleave on a GPU repeats tensor, t.item()/h.item()/w.item()
    per image/video, llm_pos_ids_list[-1].max() per span, ...). On large
    multimodal sequences this can cost tens of ms per step.

    This implementation pulls input_ids + grid_thw to host once at entry, then
    builds the (3, n_valid) position tensor for each row with numpy
    slice-assigns (avoiding per-token list.append + torch.tensor() conversion),
    and copies it back with a single H2D per row. The trivial no-vision
    branch is unchanged.
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id

    has_vision = input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None)
    if not has_vision:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas

    device = input_ids.device
    dtype = input_ids.dtype

    # One-shot device -> host pull. After this point everything is plain
    # numpy / Python int — no further device syncs inside the hot loop.
    input_ids_np = input_ids.detach().cpu().numpy()
    if attention_mask is not None:
        attention_mask_np = attention_mask.detach().to(input_ids.device).cpu().numpy()
    else:
        attention_mask_np = np.ones_like(input_ids_np)

    if image_grid_thw is not None:
        image_thw_np = image_grid_thw.detach().cpu().numpy()
    else:
        image_thw_np = np.zeros((0, 3), dtype=np.int64)
    if video_grid_thw is not None:
        # Mirror the upstream repeat_interleave on T axis + set T=1: each
        # frame of a video is treated as an independent (1, H, W) entry so
        # video timestamps carry the temporal position.
        video_thw_np = video_grid_thw.detach().cpu().numpy()
        if video_thw_np.shape[0] > 0:
            video_thw_np = np.repeat(video_thw_np, video_thw_np[:, 0], axis=0)
            video_thw_np[:, 0] = 1
    else:
        video_thw_np = np.zeros((0, 3), dtype=np.int64)

    B, S = input_ids.shape
    position_ids = torch.ones(3, B, S, dtype=dtype, device=device)
    mrope_position_deltas: list[int] = []

    image_index = 0
    video_index = 0
    for i in range(B):
        row = input_ids_np[i]
        mask_row_bool = attention_mask_np[i].astype(bool)
        valid_tokens = row[mask_row_bool]
        n_valid = int(valid_tokens.shape[0])

        # Vectorized scan: find all <vision_start> followed by image_pad or video_pad.
        vstart_positions = np.flatnonzero(valid_tokens == vision_start_token_id)
        vstart_positions = vstart_positions[vstart_positions + 1 < n_valid]
        next_tokens = valid_tokens[vstart_positions + 1]
        is_image = next_tokens == image_token_id
        is_video = next_tokens == video_token_id
        keep = is_image | is_video
        vstart_positions = vstart_positions[keep]
        is_video_kept = is_video[keep]
        n_spans = int(vstart_positions.shape[0])

        out = np.empty((3, n_valid), dtype=np.int64)
        st = 0
        last_max = -1  # so st_idx = last_max + 1 starts at 0
        for s_idx in range(n_spans):
            # span_start is the first vision token after <vision_start>.
            span_start = int(vstart_positions[s_idx]) + 1
            if is_video_kept[s_idx]:
                t, h, w = video_thw_np[video_index]
                video_index += 1
            else:
                t, h, w = image_thw_np[image_index]
                image_index += 1
            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size
            text_len = span_start - st
            st_idx = last_max + 1
            base = text_len + st_idx

            if text_len > 0:
                run = np.arange(st_idx, st_idx + text_len, dtype=np.int64)
                out[0, st : st + text_len] = run
                out[1, st : st + text_len] = run
                out[2, st : st + text_len] = run

            v_len = llm_grid_t * llm_grid_h * llm_grid_w
            v_start = st + text_len
            if v_len > 0:
                t_axis = np.repeat(np.arange(llm_grid_t, dtype=np.int64), llm_grid_h * llm_grid_w) + base
                h_axis = np.tile(np.repeat(np.arange(llm_grid_h, dtype=np.int64), llm_grid_w), llm_grid_t) + base
                w_axis = np.tile(np.arange(llm_grid_w, dtype=np.int64), llm_grid_t * llm_grid_h) + base
                out[0, v_start : v_start + v_len] = t_axis
                out[1, v_start : v_start + v_len] = h_axis
                out[2, v_start : v_start + v_len] = w_axis

            st = v_start + v_len
            last_max = base + max(llm_grid_t, llm_grid_h, llm_grid_w) - 1

        if st < n_valid:
            text_len = n_valid - st
            st_idx = last_max + 1
            run = np.arange(st_idx, st_idx + text_len, dtype=np.int64)
            out[0, st:n_valid] = run
            out[1, st:n_valid] = run
            out[2, st:n_valid] = run
            last_max = st_idx + text_len - 1

        # Single H2D for this row's positions.
        llm_positions = torch.from_numpy(out).to(device=device, dtype=dtype, non_blocking=True)
        if attention_mask is not None:
            position_ids[:, i, attention_mask[i] == 1] = llm_positions
        else:
            position_ids[:, i, :] = llm_positions
        mrope_position_deltas.append(last_max + 1 - int(S))

    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=device, dtype=dtype).unsqueeze(1)
    return position_ids, mrope_position_deltas


def qwen2_5_vl_rope_index(
    self: Qwen2_5_VLModel,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    spatial_merge_size = self.config.vision_config.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    vision_start_token_id = self.config.vision_start_token_id
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is not None:
            attention_mask = attention_mask == 1
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            if attention_mask is not None:
                input_ids = input_ids[attention_mask[i]]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                ## normalize type, send to device.
                second_per_grid_t = torch.as_tensor(
                    second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
                )

                time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[..., i, attention_mask[i]] = llm_positions.to(position_ids.device)
            else:
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas).unsqueeze(1).to(device=input_ids.device)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
