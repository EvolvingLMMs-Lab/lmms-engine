import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor


class Qwen3OmniMoeExperts(nn.Module):
    """
    Stacked expert implementation for Qwen3-Omni MoE models.

    Instead of using nn.ModuleList with separate expert modules, this class stacks
    all expert weights into single 3D tensors. This approach provides:
    - Reduced module overhead
    - Efficient batched expert computation
    - Seamless integration with DTensor for expert parallelism (EP)
    - Better memory locality

    Weight shapes:
        gate_proj: [num_experts, intermediate_size, hidden_dim]
        up_proj: [num_experts, intermediate_size, hidden_dim]
        down_proj: [num_experts, hidden_dim, intermediate_size]

    With expert parallelism enabled, these become DTensors sharded along dimension 0
    (the expert dimension), distributing experts across multiple GPUs.

    Args:
        num_experts: Total number of experts in the MoE layer
        hidden_dim: Hidden dimension size of the model
        intermediate_size: Intermediate dimension size for expert MLPs
        act_fn: Activation function for the gating mechanism (typically SiLU)

    Example:
        >>> experts = Qwen3OmniMoeExperts(60, 2048, 5632, nn.SiLU())
        >>> # With EP=4, each GPU holds 15 experts
        >>> # Input: List of tensors, one per expert with shape [tokens_for_expert_i, 2048]
        >>> # Output: Concatenated tensor with shape [total_tokens, 2048]
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_size: int,
        act_fn: nn.Module,
    ):
        super().__init__()

        # Initialize stacked expert parameters
        # Shape: [num_experts, intermediate_size, hidden_dim]
        self.gate_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )

        # Shape: [num_experts, intermediate_size, hidden_dim]
        self.up_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_dim),
            requires_grad=True,
        )

        # Shape: [num_experts, hidden_dim, intermediate_size]
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_size),
            requires_grad=True,
        )

        self.num_experts = num_experts
        self.act_fn = act_fn

    def forward(self, *routed_input):
        """
        Forward pass through local experts with SwiGLU activation.

        This method processes tokens through each expert sequentially. Accepts variable
        number of arguments to handle both EP and non-EP cases.

        Args:
            *routed_input: Variable number of tensors, one per local expert.
                Each tensor has shape [num_tokens_for_this_expert, hidden_dim].
                When called with EP enabled, ParallelStyle._input_fn pre-splits the input
                into a tuple of tensors which gets unpacked as separate arguments.

        Returns:
            torch.Tensor: Concatenated expert outputs with shape
                [total_local_tokens, hidden_dim], where total_local_tokens is the
                sum of tokens across all local experts.

        Expert Computation (SwiGLU):
            For each expert i and its input x:
            1. hidden = act_fn(x @ gate_proj[i].T)      # Gating pathway
            2. hidden = hidden * (x @ up_proj[i].T)     # Element-wise with up pathway
            3. output = hidden @ down_proj[i].T          # Project back to hidden_dim

        DTensor Handling:
            When using expert parallelism, expert parameters are DTensors (distributed
            tensors). We convert to local tensors before computation since the routing
            has already distributed tokens to the correct GPU.

        Performance Notes:
            - Each expert is processed independently (could be parallelized further)
            - Memory efficient: only processes local tokens
            - Gradient computation works transparently with DTensor
        """
        out_experts_split = []

        # Convert DTensor to local tensors if using expert parallelism
        # DTensor.to_local() retrieves the local shard on this GPU
        if isinstance(self.down_proj, DTensor):
            down_proj = self.down_proj.to_local()
            up_proj = self.up_proj.to_local()
            gate_proj = self.gate_proj.to_local()
        else:
            down_proj = self.down_proj
            up_proj = self.up_proj
            gate_proj = self.gate_proj

        # Process each expert independently
        for idx, x in enumerate(routed_input):
            # SwiGLU: Gated Linear Unit with SiLU activation
            # Step 1: Gating pathway with activation
            # x: [num_tokens, hidden_dim], gate_proj[idx]: [intermediate_size, hidden_dim]
            # Result: [num_tokens, intermediate_size]
            hidden = self.act_fn(torch.matmul(x, gate_proj[idx].transpose(-2, -1)))

            # Step 2: Element-wise multiply with up projection pathway
            # up_proj[idx]: [intermediate_size, hidden_dim]
            # Result: [num_tokens, intermediate_size]
            hidden = hidden * torch.matmul(x, up_proj[idx].transpose(-2, -1))

            # Step 3: Down projection back to hidden dimension
            # down_proj[idx]: [hidden_dim, intermediate_size]
            # Result: [num_tokens, hidden_dim]
            hidden = torch.matmul(hidden, down_proj[idx].transpose(-2, -1))

            out_experts_split.append(hidden)

        # Concatenate outputs from all local experts
        # Shape: [total_local_tokens, hidden_dim]
        return torch.cat(out_experts_split, dim=0)
