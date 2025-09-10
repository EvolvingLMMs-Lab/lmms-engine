from collections import deque
from typing import Protocol

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.distributed import all_gather_into_tensor, gather, scatter
from torch.distributed.tensor import DTensor

from lmms_engine.utils.logging_utils import Logging


def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_SIZE_M": blk_m, "BLOCK_SIZE_K": blk_k, "GROUP_SIZE_M": grp_sz},
            num_stages=n_stages,
            num_warps=n_warps,
        )
        for blk_m in [32, 64, 128]
        for blk_k in [32, 64]
        for grp_sz in [8]
        for n_stages in [3, 4, 5]
        for n_warps in [4, 8]
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["M", "K"],
)
@triton.jit
def mmt_kernel(
    x,
    y,
    M,
    K,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Core kernel jit function of matmul_transpose that computes y = x @ x.T
    The code is a simple adaptation from the triton `matmul` tutorial:
    https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if pid_m > pid_n:
        return

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # we use a & b ptrs to denote different rows of x.
    a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_xk
    # use dtype.element_ty to accomodate different input datatypes as in cpp templates
    # https://github.com/triton-lang/triton/issues/2252
    c = accumulator.to(x.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)

    # transpose and copy
    if pid_m < pid_n:
        ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def matmul_transpose_assign(d_in, d_out):
    assert d_in.is_cuda, "Input `d_in` must be a CUDA tensor"
    assert d_out.is_cuda, "Input `d_out` must be a CUDA tensor"
    assert (
        d_in.device == d_out.device
    ), "Inputs `d_in` and `d_out` must be on the same CUDA device"
    assert d_in.dtype == d_out.dtype, "Inputs must have the same data type"
    assert d_in.ndim == 2, "Input `d_in` must be a 2D tensor"
    assert d_out.ndim == 2, "Input `d_out` must be a 2D tensor"
    assert (
        d_in.size(0) == d_out.size(0) == d_out.size(0)
    ), "First dimension of `d_in` must match first and second dimension of `d_out`"

    d_in = d_in.contiguous()
    M, K = d_in.shape
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_M"]),
    )
    with torch.cuda.device(d_in.device.index):
        mmt_kernel[grid](
            d_in,
            d_out,
            M,
            K,
            d_in.stride(0),
            d_in.stride(1),
            d_out.stride(0),
            d_out.stride(1),
        )


def matmul_transpose(d_in):
    M, _ = d_in.shape
    d_out = torch.empty((M, M), device=d_in.device, dtype=d_in.dtype)
    matmul_transpose_assign(d_in, d_out)
    return d_out


def fast_newtonschulz(G: Tensor, steps: int = 5) -> Tensor:
    """
    adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
    Arguments:
        G: The gradient or momentum matrix to be orthogonalized.
        steps: Number of Newton-Schulz iterations.
    """
    assert G.ndim >= 2, f"Gradient must be a 2D tensor, but got {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    buf1 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    buf2 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        matmul_transpose_assign(X, buf1)
        matmul_transpose_assign(buf1, buf2)
        B = b * buf1 + c * buf2
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def apply_momentum(grad, momentum, beta, nesterov):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    return update


def apply_scaling(grad, rms_scale=False):
    if rms_scale:
        # https://github.com/MoonshotAI/Moonlight/blob/5afcb6911077e7f182d05865fe90d9f39abcbcbd/examples/toy_train.py#L146
        grad *= 0.2 * (max(grad.shape[1], grad.shape[0])) ** 0.5
        return grad
    else:
        # https://github.com/KellerJordan/Muon/blob/f90a42b28e00b8d9d2d05865fe90d9f39abcbcbd/muon.py#L40
        grad *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
        return grad


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class Work(Protocol):
    def __init__(self, param, state, group, index: int):
        ...

    def start(self):
        ...

    def finish(self):
        ...


class Fsdp1dWork:
    """
    muon handle for fsdp2 1d mesh.
    """

    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group
        self.index = index

        self._intermediate_state = None

    def start(self):
        self.param.grad = apply_momentum(
            self.param.grad,
            self.state["momentum_buffer"],
            self.group["momentum"],
            self.group["nesterov"],
        )
        grad = self.param.grad
        assert isinstance(grad, DTensor), "only supports DTensor parameters"
        assert grad.device_mesh.ndim == 1, "only supports 1D mesh"

        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()

        dest_rank = self.index % world_size
        if rank == dest_rank:
            gather_lists = [
                torch.zeros_like(input=grad.to_local()) for _ in range(world_size)
            ]
            gather_handle = gather(
                grad.to_local(),
                gather_lists,
                group_dst=dest_rank,
                group=pg,
                async_op=True,
            )

        else:
            gather_lists = None
            gather_handle = gather(
                grad.to_local(), None, group_dst=dest_rank, group=pg, async_op=True
            )

        self._intermediate_state = [dest_rank, gather_handle, gather_lists]

    def finish(self):
        assert self._intermediate_state is not None, "gather work must be called first"
        grad = self.param.grad
        rank = grad.device_mesh.get_rank()
        world_size = grad.device_mesh.size()
        pg = grad.device_mesh.get_group()
        dest_rank, gather_handle, gather_lists = self._intermediate_state
        gather_handle.wait()
        if rank == dest_rank:
            g_full_block = torch.cat(gather_lists, dim=0)
            g_full_block.copy_(fast_newtonschulz(g_full_block, self.group["ns_steps"]))
            g_full_block = g_full_block.type_as(grad)
            chunks = list(g_full_block.chunk(chunks=world_size, dim=0))
            scatter(
                grad.to_local(),
                scatter_list=chunks,
                src=dest_rank,
                group=pg,
                async_op=False,
            )
        else:
            scatter(grad.to_local(), None, src=dest_rank, group=pg, async_op=False)
        update = apply_scaling(grad, self.group["rms_scale"])
        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])


class TpFsdp2dWork:
    """
    Muon work for TP + FSDP mesh
    """

    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("not implemented")


class EpFsdp2dWork:
    """
    Muon work for EP mesh
    """

    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("not implemented")


class TpEpFsdp3dWork:
    """
    Muon work for TP + EP mesh
    """

    def __init__(self, param, state, group, index: int):
        raise NotImplementedError("not implemented")


class SingelDeviceWork:
    """
    muon handle for single device.
    """

    def __init__(self, param, state, group, index: int):
        self.param = param
        self.state = state
        self.group = group

    def start(self):
        update = muon_update(
            self.param.grad,
            self.state["momentum_buffer"],
            self.group["momentum"],
            self.group["nesterov"],
            self.group["ns_steps"],
            self.group["rms_scale"],
        )
        self.param.mul_(1 - self.group["lr"] * self.group["weight_decay"])
        self.param.add_(update.reshape(self.param.shape), alpha=-self.group["lr"])

    def finish(self):
        pass


class Muon(torch.optim.Optimizer):
    def __init__(
        self, param_groups, defaults=dict(lr=0.02), is_deepspeed_enabled=False
    ):
        self.is_deepspeed_enabled = is_deepspeed_enabled
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                rms_scale = group.get("rms_scale", True)
                if self.is_deepspeed_enabled and rms_scale:
                    Logging.warning(
                        "rms_scale for Muon is not supported in deepspeed, setting rms_scale to False"
                    )
                    rms_scale = False
                group["rms_scale"] = rms_scale
                nesterov = group.get("nesterov", True)
                if self.is_deepspeed_enabled and not nesterov:
                    Logging.warning(
                        "disabled nesterov for Muon is not supported in deepspeed, setting nesterov to True"
                    )
                    nesterov = False
                group["nesterov"] = nesterov
                group["ns_steps"] = group.get("ns_steps", 5)
                assert set(group.keys()) == set(
                    [
                        "params",
                        "lr",
                        "momentum",
                        "weight_decay",
                        "use_muon",
                        "rms_scale",
                        "nesterov",
                        "ns_steps",
                    ]
                )
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(
                    ["params", "lr", "betas", "eps", "weight_decay", "use_muon"]
                )
        super().__init__(param_groups, defaults=defaults)

    def _get_work_class(self, p: torch.Tensor) -> tuple[type[Work], int]:
        """
        dispatch the work class based on the mesh dimension.
        """
        if isinstance(p, DTensor):
            if p.device_mesh.ndim == 1:
                return Fsdp1dWork, 8
            elif p.device_mesh.ndim == 2:
                return TpFsdp2dWork, 8
            else:
                raise ValueError(f"Unsupported mesh dimension: {p.device_mesh.ndim}")
        else:
            return SingelDeviceWork, 1

    @torch.no_grad()
    def step(self, closure=None):
        if self.is_deepspeed_enabled:
            return self.step_deepspeed(closure)
        else:
            return self.step_fsdp(closure)

    @torch.no_grad()
    def step_deepspeed(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                # for deepspeed, if we set the param.use_muon as True, the update is done in the deepspeed's framework.
                # we only need to do the weight decay and update here.
                for p in group["params"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(p.grad.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    from deepspeed.runtime.zero.muon.original_muon import adam_update

                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss

    @torch.no_grad()
    def step_fsdp(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        dq: deque[Work] = deque()
        for group in self.param_groups:
            if group["use_muon"]:
                for i, p in enumerate(group["params"]):
                    assert p.ndim == 2, p.shape
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    class_work, prefetch_factor = self._get_work_class(p)
                    work = class_work(p, state, group, i)
                    work.start()
                    dq.append(work)
                    if len(dq) > prefetch_factor:
                        dq.popleft().finish()
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        for work in dq:
            work.finish()

        return loss
        # loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()
        # for group in self.param_groups:
        #     if group["use_muon"]:
        #         params: list[Tensor] = group["params"]
        #         update_buffer: Tensor = group["update_buffer"]
        #         update_buffer_views: list[Tensor] = group["update_buffer_views"]

        #         handle = None
        #         params_world = None
        #         weight_decays_world = None

        #         def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
        #             handle.wait()
        #             for p_world, g_world, weight_decay in zip(params_world, update_buffer_views, weight_decays_world):
        #                 alpha = -group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5
        #                 p_world.mul_(1 - group["lr"] * group["weight_decays"])
        #                 p_world.add_(
        #                     g_world.view_as(p_world),
        #                     alpha=alpha
        #                 )

        #         for base_i in range(len(params))[::self.world_size]:
        #             if base_i + self.rank < len(params):
        #                 p = params[base_i + self.rank]
        #                 g = p.grad
        #                 assert g is not None
        #                 assert p.shape == g.shape
        #                 state = self.state[p]
        #                 if "momentum_buffer" not in state:
        #                     state["momentum_buffer"] = torch.zeros_like(g)

        #                 buf: Tensor = state["momentum_buffer"]
        #                 buf.lerp_(g, 1 - group["beta"])
        #                 g = g.lerp_(buf, group["beta"]) if group["nesterov"] else buf
        #                 g = fast_newtonschulz(g, steps=group["ns_steps"]).flatten()
        #             else:
        #                 g = update_buffer_views[self.rank]
        #             if base_i > 0:
        #                 update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
        #             handle = all_gather_into_tensor(update_buffer, g, async_op=True)
        #             params_world = params[base_i : base_i + self.world_size]
        #             weight_decays_world = weight_decays[base_i : base_i + self.world_size]
        #         update_prev()
        #     else:
        #         for p in group["params"]:
        #             g = p.grad
        #             assert g is not None
        #             state = self.state[p]
        #             if "step" not in state: state["step"] = 0
        #             if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
        #             if "momentum_sq_buffer" not in state: state["momentum_sq_buffer"] = torch.zeros_like(g)
        #             m: Tensor = state["momentum_buffer"]
        #             v: Tensor = state["momentum_sq_buffer"]
        #             m.lerp_(g, 1 - group["adam_beta1"])
        #             v.mul_(group["adam_beta2"]).addcmul_(g, g, value=1.0 - group["adam_beta2"])
        #             state["step"] += 1
        #             bias_correction1 = 1.0 - group["adam_beta1"] ** state["step"]
        #             bias_correction2 = 1.0 - group["adam_beta2"] ** state["step"]
        #             step_size = group["lr"] * (bias_correction2) ** 0.5 / bias_correction1
        #             norm_grad = m / (v.sqrt().add_(group["adam_eps"]))
        #             if weight_decay > 0.0:
        #                 p.add_(p, alpha=(-group["lr"] * group["weight_decays"]))
        #             p.add_(norm_grad, alpha=-step_size)
        # return loss
