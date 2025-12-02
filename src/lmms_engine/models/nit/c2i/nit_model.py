# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
from einops import rearrange, repeat
from flash_attn import flash_attn_varlen_func
from nit.models.utils.funcs import get_parameter_dtype
from nit.models.utils.pos_embeds.rope import VisionRotaryEmbedding, rotate_half
from typing import Optional

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def build_mlp(hidden_size, projector_dim, z_dim):
    return nn.Sequential(
                nn.Linear(hidden_size, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, projector_dim),
                nn.SiLU(),
                nn.Linear(projector_dim, z_dim),
            )
#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################            
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Attention Block                               #
#################################################################################

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, cu_seqlens, freqs_cos, freqs_sin) -> torch.Tensor:
        N, C = x.shape
        qkv = self.qkv(x).reshape(N, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        ori_dtype = qkv.dtype
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        q = q * freqs_cos + rotate_half(q) * freqs_sin
        k = k * freqs_cos + rotate_half(k) * freqs_sin
        q, k = q.to(ori_dtype), k.to(ori_dtype)
        
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        x = flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen
        ).reshape(N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



#################################################################################
#                                 Core NiT Model                                #
#################################################################################

class NiTBlock(nn.Module):
    """
    A NiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=block_kwargs["qk_norm"]
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
            )
        use_adaln_lora = block_kwargs.get('use_adaln_lora', False)
        if use_adaln_lora:
            adaln_lora_dim = block_kwargs['adaln_lora_dim']
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_lora_dim, bias=True),
                nn.Linear(adaln_lora_dim, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )

    def forward(self, x, c, cu_seqlens, freqs_cos, freqs_sin):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cu_seqlens, freqs_cos, freqs_sin)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    """
    The final layer of NiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


class NiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        encoder_depth=4,
        projector_dim=2048,
        z_dim=768,
        use_checkpoint: bool = False,
        custom_freqs: str = 'normal',
        theta: int = 10000,
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
        **block_kwargs # fused_attn
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.use_checkpoint = use_checkpoint
        
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True, strict_img_size=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size) # timestep embedding type
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.rope = VisionRotaryEmbedding(
            head_dim=hidden_size//num_heads, custom_freqs=custom_freqs, theta=theta,
            max_pe_len_h=max_pe_len_h, max_pe_len_w=max_pe_len_w, decouple=decouple,
            ori_max_pe_len=ori_max_pe_len
        )

        self.projector = build_mlp(hidden_size, projector_dim, z_dim) 
        
        self.blocks = nn.ModuleList([
            NiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in NiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def get_rope(self, hw_list):
        grids = []
        for h, w in hw_list:
            grid_h = torch.arange(h)
            grid_w = torch.arange(w)
            grid = torch.meshgrid(grid_h, grid_w, indexing='xy') 
            grid = torch.stack(grid, dim=0).reshape(2, -1)
            grids.append(grid)
        grids = torch.cat(grids, dim=-1)
        freqs_cos, freqs_sin = self.rope.get_cached_2d_rope_from_grid(grids)
        return freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)

    def forward(self, x, t, y, hw_list, return_zs=False, return_logvar=False):
        """
        Forward pass of NiT.
        x: (N, C, p, p) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x)                  # (N, C, p, p) -> (N, 1, D), where T = H * W / patch_size ** 2
        x = x.squeeze(1)                        # (N, D)
        B = hw_list.shape[0]

        freqs_cos, freqs_sin = self.get_rope(hw_list)   # (N, D_h)
        seqlens = hw_list[:, 0] * hw_list[:, 1]
        cu_seqlens = torch.cat([
            torch.tensor([0], device=hw_list.device, dtype=torch.int), 
            torch.cumsum(seqlens, dim=0, dtype=torch.int)
        ])

        # timestep and class embedding
        t_embed = self.t_embedder(t)            # (B, D)
        y = self.y_embedder(y)                  # (B, D)
        c = t_embed + y                         # (B, D)
        
        # (B, D) -> (N, D)
        c = torch.cat([c[i].unsqueeze(0).repeat(seqlens[i], 1) for i in range(B)], dim=0)
        
        zs=[]
        for i, block in enumerate(self.blocks):
            if not self.use_checkpoint:
                x = block(x, c, cu_seqlens, freqs_cos, freqs_sin)   # (N, D)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    self.ckpt_wrapper(block), x, c, cu_seqlens, freqs_cos, freqs_sin
                )  
            if (i + 1) == self.encoder_depth and return_zs:
                zs = [self.projector(x)]
        x = self.final_layer(x, c)              # (N, out_channels * patch_size ** 2)
        
        # (N, out_channels * patch_size ** 2) -> (N, out_channels, p, p)
        x = rearrange(x, 'n (c p1 p2) -> n c p1 p2', p1=self.patch_size, p2=self.patch_size)                  
        if return_zs:
            return x, zs
        else:
            return x  


    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)

