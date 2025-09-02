# This monkey patch does not needed for liger == 0.6.2

# import torch
# import triton

# from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
# from liger_kernel.ops.utils import (
#     amp_custom_bwd,
#     amp_custom_fwd,
#     element_mul_kernel,
#     is_hip,
# )

# # The hard limit of TRITON_MAX_TENSOR_NUMEL is 1048576 https://github.com/triton-lang/triton/blob/ba42a5c68fd0505f8c42f4202d53be0f8d9a5fe0/python/triton/language/core.py#L19
# # However, setting limit as 65536 as in LayerNorm tutorial is faster because of less register spilling
# # The optimal maximum block size depends on your hardware, your kernel, and your dtype
# MAX_FUSED_SIZE = 65536 // 2

# def myfused_linear_cross_entropy_forward(
#     _input,
#     weight,
#     target,
#     bias=None,
#     ignore_index=-100,
#     lse_square_scale=0.0,
#     label_smoothing=0.0,
#     reduction="mean",
#     softcap=None,
# ):
#     device = _input.device

#     # inputs have shape: BT x H
#     # materialized activations will have shape: BT x V
#     # the increase in memory = BT x V
#     # reduction can be achieved by partitioning the number of tokens BT into smaller chunks.
#     # for ex: if we were to achieve the same memory consumption as BT x H, then the chunk size should be:
#     # inc_factor = (V+H-1)//H, chunk_size = (BT + inc_factor - 1)//inc_factor
#     # for ex: BT = 4096*4, V = 32000, H = 4096 ==> inc_factor = 8, chunk_size = 2048
#     BT, H = _input.shape
#     V = weight.shape[0]
#     BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

#     inc_factor = triton.cdiv(V, H)  # (V + H - 1) // H
#     chunk_size = triton.next_power_of_2(
#         triton.cdiv(BT, inc_factor)
#     )  # (BT + inc_factor - 1) // inc_factor
#     num_chunks = triton.cdiv(BT, chunk_size)  # (BT + chunk_size - 1) // chunk_size

#     grad_weight = (
#         torch.zeros_like(weight, device=device) if weight.requires_grad else None
#     )
#     grad_input = torch.zeros_like(_input, device=device)
#     grad_bias = torch.zeros_like(bias, device=device) if bias is not None else None
#     # we use fp32 for loss accumulator
#     loss_1d = torch.zeros(BT, dtype=torch.float32, device=device)

#     # NOTE: skip .item() here to avoid CUDA synchronization
#     total_n_non_ignore = (target != ignore_index).sum()

#     for chunk_id in range(num_chunks):
#         start_idx = chunk_id * chunk_size
#         end_idx = min((chunk_id + 1) * chunk_size, BT)
#         _input_chunk = _input[start_idx:end_idx]  # chunk_size x H

#         # when doing matmul, use the original precision
#         logits_chunk = _input_chunk @ weight.t()  # chunk_size x V
#         if bias is not None:
#             logits_chunk = logits_chunk + bias
#         target_chunk = target[start_idx:end_idx]  # chunk_size,

#         n_rows = logits_chunk.shape[0]

#         # unreduced loss
#         loss_1d_slice = loss_1d[start_idx:end_idx]  # chunk_size,
#         n_non_ignore = (target_chunk != ignore_index).sum().item()

#         # ensure _input and target are contiguous
#         logits_chunk = logits_chunk.contiguous()
#         target_chunk = target_chunk.contiguous()

#         # Here we calculate the gradient of logits_chunk in place so we can save memory.
#         liger_cross_entropy_kernel[(n_rows,)](
#             X_ptr=logits_chunk,
#             X_stride=logits_chunk.stride(-2),
#             Y_ptr=target_chunk,
#             Y_stride=target_chunk.stride(-1),  # always 1
#             loss_ptr=loss_1d_slice,
#             z_loss_ptr=loss_1d_slice,  # dummy ptr, not used
#             loss_stride=loss_1d_slice.stride(-1),  # always 1
#             n_cols=V,
#             n_non_ignore=n_non_ignore,
#             ignore_index=ignore_index,
#             lse_square_scale=lse_square_scale,
#             label_smoothing=label_smoothing,
#             reduction=reduction,
#             softcap=softcap if softcap is not None else 0.0,
#             RETURN_Z_LOSS=0,  # False
#             HAS_SOFTCAPPING=True if softcap is not None else False,
#             BLOCK_SIZE=BLOCK_SIZE,
#             num_warps=32 if not is_hip() else 16,
#         )

#         # gradient of logits_chunk is computed in-place by the above triton kernel and is of shape: chunk_size x V
#         # thus grad_input[start_idx: end_idx] should be of shape: chunk_size x H
#         # additionally, since we are chunking the inputs, observe that the loss and gradients are calculated only
#         # on `n_non_ignore` tokens. However, the gradient of the input should be calculated for all tokens.
#         # Thus, we need an additional scaling factor of (n_non_ignore/total_n_non_ignore) to scale the gradients.

#         if reduction == "mean":
#             alpha = n_non_ignore / total_n_non_ignore if total_n_non_ignore > 0 else 0.0
#         else:
#             alpha = 1.0

#         loss_1d[start_idx:end_idx] = loss_1d_slice * alpha
#         grad_logits_chunk = logits_chunk * alpha  # chunk_size x V

#         grad_input[start_idx:end_idx] = grad_logits_chunk @ weight

#         if grad_weight is not None:
#             torch.addmm(
#                 input=grad_weight,
#                 mat1=logits_chunk.t(),
#                 mat2=_input_chunk,
#                 out=grad_weight,
#                 alpha=alpha,
#                 beta=1.0,
#             )

#         if bias is not None:
#             torch.add(
#                 input=grad_bias,
#                 other=logits_chunk.sum(dim=0),
#                 out=grad_bias,
#                 alpha=alpha,
#             )

#     loss = torch.sum(loss_1d)
#     if reduction == "none":
#         loss = loss_1d
#     return loss, grad_input, grad_weight, grad_bias


# from liger_kernel.ops import fused_linear_cross_entropy

# fused_linear_cross_entropy.fused_linear_cross_entropy_forward = myfused_linear_cross_entropy_forward


import torch
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction

from typing import Optional

class MyLigerFusedLinearCrossEntropyLoss(torch.nn.Module):
    def __init__(
        self,
        ce_weight: Optional[torch.FloatTensor] = None,
        ignore_index: int = -100,
        lse_square_scale: float = 0.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
        softcap: Optional[float] = None,
        return_z_loss: bool = False,
        accum_dtype: Optional[torch.dtype] = None,
        use_token_scaling: bool = False,
    ):
        super().__init__()
        assert (label_smoothing >= 0) and (label_smoothing <= 1), (
            f"label_smoothing must be between 0.0 and 1.0. Got: {label_smoothing}"
        )
        assert reduction in {
            "mean",
            "sum",
            "none"
        }, f"reduction must be 'mean' or 'sum' or 'none'. Got: {reduction}"
        assert softcap is None or softcap > 0, f"softcap must greater than 0.0 or None. Got: {softcap}"
        self.ce_weight = ce_weight
        self.ignore_index = ignore_index
        self.lse_square_scale = lse_square_scale
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.softcap = softcap
        self.return_z_loss = return_z_loss
        self.accum_dtype = accum_dtype
        self.use_token_scaling = use_token_scaling

    def forward(self, lin_weight, _input, target, bias=None):
        loss, z_loss = LigerFusedLinearCrossEntropyFunction.apply(
            _input,
            lin_weight,
            target,
            bias,
            self.ce_weight,
            self.ignore_index,
            self.lse_square_scale,
            self.label_smoothing,
            self.reduction,
            self.softcap,
            self.return_z_loss,
            self.accum_dtype,
            self.use_token_scaling,
        )
        if not self.return_z_loss:
            return loss
        return loss, z_loss