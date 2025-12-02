# --------------------------------------------------------
# FiT: A Flexible Vision Transformer for Image Generation
#
# Based on the following repository
# https://github.com/lucidrains/rotary-embedding-torch
# https://github.com/jquesnelle/yarn/blob/HEAD/scaled_rope
# https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=b80b3f37
# --------------------------------------------------------

import math
from math import pi
from typing import Optional, Any, Union, Tuple
import torch
from torch import nn

from einops import rearrange, repeat
from functools import lru_cache

#################################################################################
#                                 NTK Operations                                #
#################################################################################

def find_correction_factor(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base)) #Inverse dim formula to find number of rotations

def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_factor(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_factor(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1) #Clamp values just in case

def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001 #Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def find_newbase_ntk(dim, base=10000, scale=1):
    # Base change formula
    return base * scale ** (dim / (dim-2))

def get_mscale(scale=torch.Tensor):
    # if scale <= 1:
    #     return 1.0
    # return 0.1 * math.log(scale) + 1.0
    return torch.where(scale <= 1., torch.tensor(1.0), 0.1 * torch.log(scale) + 1.0)

def get_proportion(L_test, L_train):
    L_test = L_test * 2
    return torch.where(torch.tensor(L_test/L_train) <= 1., torch.tensor(1.0), torch.sqrt(torch.log(torch.tensor(L_test))/torch.log(torch.tensor(L_train))))
    # return torch.sqrt(torch.log(torch.tensor(L_test))/torch.log(torch.tensor(L_train)))



#################################################################################
#                                 Rotate Q or K                                 #
#################################################################################

def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    return rearrange(x, '... d r -> ... (d r)')



#################################################################################
#                               Core Vision RoPE                                #
#################################################################################

class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,  # embed dimension for each head
        custom_freqs: str = 'normal',
        theta: int = 10000,
        online_rope: bool = False,
        max_cached_len: int = 1024,
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
    ):
        super().__init__()
        
        dim = head_dim // 2
        assert dim % 2 == 0 # accually, this is important
        self.dim = dim
        self.custom_freqs = custom_freqs.lower()
        self.theta = theta
        self.decouple = decouple
        self.ori_max_pe_len = ori_max_pe_len
        
        self.custom_freqs = custom_freqs.lower()
        if not online_rope:
            if self.custom_freqs in ['normal', 'scale1', 'scale2']:
                freqs_h = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
                freqs_w = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
            else:
                if decouple:
                    freqs_h = self.get_1d_rope_freqs(theta, dim, max_pe_len_h, ori_max_pe_len)
                    freqs_w = self.get_1d_rope_freqs(theta, dim, max_pe_len_w, ori_max_pe_len)
                else:
                    max_pe_len = max(max_pe_len_h, max_pe_len_w)
                    freqs_h = self.get_1d_rope_freqs(theta, dim, max_pe_len, ori_max_pe_len)
                    freqs_w = self.get_1d_rope_freqs(theta, dim, max_pe_len, ori_max_pe_len)
            
            self.register_buffer('freqs_h', freqs_h, persistent=False)        
            self.register_buffer('freqs_w', freqs_w, persistent=False)        
            
            if max_pe_len_h != None and max_pe_len_w != None and ori_max_pe_len != None:
                attn_factor = 1.0
                scale = torch.clamp_min(torch.tensor(max(max_pe_len_h, max_pe_len_w)) / ori_max_pe_len, 1.0)   # dynamic scale
                self.mscale = get_mscale(scale).to(scale) * attn_factor # Get n-d magnitude scaling corrected for interpolation
                self.proportion1 = get_proportion(max(max_pe_len_h, max_pe_len_w), ori_max_pe_len)
                self.proportion2 = get_proportion(max_pe_len_h * max_pe_len_w, ori_max_pe_len ** 2)
                
                
            freqs_h_cached = torch.einsum('..., f -> ... f', torch.arange(max_cached_len), self.freqs_h)
            freqs_h_cached = repeat(freqs_h_cached, '... n -> ... (n r)', r = 2)
            self.register_buffer('freqs_h_cached', freqs_h_cached, persistent=False) 
            freqs_w_cached = torch.einsum('..., f -> ... f', torch.arange(max_cached_len), self.freqs_w)
            freqs_w_cached = repeat(freqs_w_cached, '... n -> ... (n r)', r = 2)
            self.register_buffer('freqs_w_cached', freqs_w_cached, persistent=False) 
        

    def get_1d_rope_freqs(self, theta, dim, max_pe_len, ori_max_pe_len):
        # scaling operations for extrapolation
        assert isinstance(ori_max_pe_len, int)
        # scale = max_pe_len / ori_max_pe_len
        if not isinstance(max_pe_len, torch.Tensor):
            max_pe_len = torch.tensor(max_pe_len)
        scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)   # dynamic scale
        
        if self.custom_freqs == 'linear': # equal to position interpolation
            freqs = 1. / torch.einsum('..., f -> ... f', scale, theta ** (torch.arange(0, dim, 2).float() / dim))
        elif self.custom_freqs == 'ntk-aware' or self.custom_freqs == 'ntk-aware-pro1' or self.custom_freqs == 'ntk-aware-pro2':
            freqs = 1. / torch.pow(
                find_newbase_ntk(dim, theta, scale).view(-1, 1), 
                (torch.arange(0, dim, 2).to(scale).float() / dim)
            ).squeeze()
        elif self.custom_freqs == 'ntk-by-parts':
            #Interpolation constants found experimentally for LLaMA (might not be totally optimal though)
            #Do not change unless there is a good reason for doing so!
            beta_0 = 1.25
            beta_1 = 0.75
            gamma_0 = 16
            gamma_1 = 2
            ntk_factor = 1
            extrapolation_factor = 1

            #Three RoPE extrapolation/interpolation methods
            freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
            freqs_linear = 1.0 / torch.einsum('..., f -> ... f', scale, (theta ** (torch.arange(0, dim, 2).to(scale).float() / dim)))
            freqs_ntk = 1. / torch.pow(
                find_newbase_ntk(dim, theta, scale).view(-1, 1), 
                (torch.arange(0, dim, 2).to(scale).float() / dim)
            ).squeeze()
            
            #Combine NTK and Linear
            low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
            freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(scale)) * ntk_factor
            freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask
            
            #Combine Extrapolation and NTK and Linear
            low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
            freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(scale)) * extrapolation_factor
            freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask
            
        elif self.custom_freqs == 'yarn':
            #Interpolation constants found experimentally for LLaMA (might not be totally optimal though)
            #Do not change unless there is a good reason for doing so!
            beta_fast = 32
            beta_slow = 1
            extrapolation_factor = 1
            
            freqs_extrapolation = 1.0 / (theta ** (torch.arange(0, dim, 2).to(scale).float() / dim))
            freqs_interpolation = 1.0 / torch.einsum('..., f -> ... f', scale, (theta ** (torch.arange(0, dim, 2).to(scale).float() / dim)))

            low, high = find_correction_range(beta_fast, beta_slow, dim, theta, ori_max_pe_len)
            freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(scale).float()) * extrapolation_factor # Get n-d rotational scaling corrected for extrapolation
            freqs = freqs_interpolation * (1 - freqs_mask) + freqs_extrapolation * freqs_mask            
        else:
            raise ValueError(f'Unknown modality {self.custom_freqs}. Only support normal, linear, ntk-aware, ntk-by-parts, yarn!')
        return freqs


    def online_get_2d_rope_from_grid(self, grid, size):
        '''
        grid: (B, 2, N)
            N = H * W
            the first dimension represents width, and the second reprensents height
            e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                    [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
        size: (B, 1, 2), h goes first and w goes last
        '''
        size = size.squeeze()   # (B, 1, 2) -> (B, 2)
        if self.decouple:
            size_h = size[:, 0]
            size_w = size[:, 1]
            freqs_h = self.get_1d_rope_freqs(self.theta, self.dim, size_h, self.ori_max_pe_len)
            freqs_w = self.get_1d_rope_freqs(self.theta, self.dim, size_w, self.ori_max_pe_len)
        else:
            size_max = torch.max(size[:, 0], size[:, 1])
            freqs_h = self.get_1d_rope_freqs(self.theta, self.dim, size_max, self.ori_max_pe_len)
            freqs_w = self.get_1d_rope_freqs(self.theta, self.dim, size_max, self.ori_max_pe_len)
        freqs_w = grid[:, 0][..., None] * freqs_w[:, None, :]
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)
        
        freqs_h = grid[:, 1][..., None] * freqs_h[:, None, :]
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)
        
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)   # (B, N, D)
        
        if self.custom_freqs == 'yarn':
            freqs_cos = freqs.cos() * self.mscale[:, None, None]
            freqs_sin = freqs.sin() * self.mscale[:, None, None]
        elif self.custom_freqs == 'ntk-aware-pro1':
            freqs_cos = freqs.cos() * self.proportion1[:, None, None]
            freqs_sin = freqs.sin() * self.proportion1[:, None, None]
        elif self.custom_freqs == 'ntk-aware-pro2':
            freqs_cos = freqs.cos() * self.proportion2[:, None, None]
            freqs_sin = freqs.sin() * self.proportion2[:, None, None]
        else:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
            
        return freqs_cos, freqs_sin  

    @lru_cache()
    def get_2d_rope_from_grid(self, grid):
        '''
        grid: (B, 2, N)
            N = H * W
            the first dimension represents width, and the second reprensents height
            e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                    [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
        '''  
        freqs_h = torch.einsum('..., f -> ... f', grid[:, 0], self.freqs_h)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)
        freqs_w = torch.einsum('..., f -> ... f', grid[:, 1], self.freqs_w)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)
        
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)   # (B, N, D)
        
        if self.custom_freqs == 'yarn':
            freqs_cos = freqs.cos() * self.mscale
            freqs_sin = freqs.sin() * self.mscale
        elif self.custom_freqs in ['ntk-aware-pro1', 'scale1']:
            freqs_cos = freqs.cos() * self.proportion1
            freqs_sin = freqs.sin() * self.proportion1
        elif self.custom_freqs in ['ntk-aware-pro2', 'scale2']:
            freqs_cos = freqs.cos() * self.proportion2
            freqs_sin = freqs.sin() * self.proportion2
        else:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()

        return freqs_cos, freqs_sin
    
    @lru_cache()
    def get_cached_2d_rope_from_grid(self, grid: torch.Tensor):
        '''
        grid: (B, 2, N)
            N = H * W
            the first dimension represents width, and the second reprensents height
            e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                    [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
        '''  
        if len(grid.shape) == 3:    # (B, 2, N)
            freqs_h, freqs_w = self.freqs_h_cached[grid[:, 0]], self.freqs_w_cached[grid[:, 1]]
        elif len(grid.shape) == 2:  # (2, N)
            freqs_h, freqs_w = self.freqs_h_cached[grid[0]], self.freqs_w_cached[grid[1]]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)   # (B, N, D)
        
        if self.custom_freqs == 'yarn':
            freqs_cos = freqs.cos() * self.mscale
            freqs_sin = freqs.sin() * self.mscale
        elif self.custom_freqs in ['ntk-aware-pro1', 'scale1']:
            freqs_cos = freqs.cos() * self.proportion1
            freqs_sin = freqs.sin() * self.proportion1
        elif self.custom_freqs in ['ntk-aware-pro2', 'scale2']:
            freqs_cos = freqs.cos() * self.proportion2
            freqs_sin = freqs.sin() * self.proportion2
        else:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
        
        return freqs_cos, freqs_sin

    @lru_cache()
    def get_cached_21d_rope_from_grid(self, grid: torch.Tensor): # for 3d rope formulation 2 !
        '''
        grid: (B, 3, N)
            N = H * W * T
            the first dimension represents width, and the second reprensents height, and the third reprensents time
            e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                    [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
                    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        '''   
        freqs_w, freqs_h = self.freqs_w_cached[grid[:, 0]+grid[:, 2]], self.freqs_h_cached[grid[:, 1]+grid[:, 2]]
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)   # (B, N, D)
        
        if self.custom_freqs == 'yarn':
            freqs_cos = freqs.cos() * self.mscale
            freqs_sin = freqs.sin() * self.mscale
        elif self.custom_freqs == 'ntk-aware-pro1':
            freqs_cos = freqs.cos() * self.proportion1
            freqs_sin = freqs.sin() * self.proportion1
        elif self.custom_freqs == 'ntk-aware-pro2':
            freqs_cos = freqs.cos() * self.proportion2
            freqs_sin = freqs.sin() * self.proportion2
        else:
            freqs_cos = freqs.cos()
            freqs_sin = freqs.sin()
        
        return freqs_cos, freqs_sin

    def forward(self, x, grid): 
        '''
        x: (B, n_head, N, D)
        grid: (B, 2, N)
        '''
        # freqs_cos, freqs_sin = self.get_2d_rope_from_grid(grid)
        # freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        # using cache to accelerate, this is the same with the above codes:
        freqs_cos, freqs_sin = self.get_cached_2d_rope_from_grid(grid)
        freqs_cos, freqs_sin = freqs_cos.unsqueeze(1), freqs_sin.unsqueeze(1)
        return  x * freqs_cos + rotate_half(x) * freqs_sin
    
 