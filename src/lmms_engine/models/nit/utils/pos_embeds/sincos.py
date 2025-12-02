#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# modified from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

import torch
import numpy as np
from einops import rearrange
import torch.nn.functional as F



def get_2d_sincos_pos_embed(embed_dim, h, w, frac_coord_size=None, scale_ratio=1.0, cls_token=False, extra_tokens=0):
    """
    args:
        h / w: int of the grid height / width
        frac_coord_size: 
            if frac_coord_size != None: 
                fractional coordinates for positional embedding is used
            else: 
                absolute coordinates for positional embedding is used
    return:
        pos_embed: [h*w, embed_dim] or [1+h*w, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')    # here w goes first
    grid = torch.stack(grid, dim=0)
    grid = rearrange(grid, '... -> 1 ...')  # (1, 2, h*w)
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(
        grid, embed_dim, frac_coord_size, scale_ratio
    )  # 1, L, D
    if cls_token and extra_tokens > 0:
        pos_embed = torch.cat([torch.zeros((1, extra_tokens, embed_dim)), pos_embed], dim=1)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(grid, embed_dim, frac_coord_size=None, scale_ratio=1.0):
    '''
    grid: (B, 2, N)
        N = H * W
        the first dimension represents width, and the second reprensents height
        e.g.,   [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
    frac_coord_size: 
        if frac_coord_size != None: 
            fractional coordinates for positional embedding is used
        else: 
            absolute coordinates for positional embedding is used
    '''
    assert embed_dim % 2 == 0
    grid = grid.float()
    if frac_coord_size != None:
        assert isinstance(frac_coord_size, (int, float))
        grid_w = grid[:, 0] / torch.max(grid[:, 0]) * frac_coord_size
        grid_h = grid[:, 1] / torch.max(grid[:, 1]) * frac_coord_size
    else:
        grid_w, grid_h = grid[:, 0]*scale_ratio, grid[:, 1]*scale_ratio
    # use half of dimensions to encode grid_h
    emb_w = get_1d_sincos_pos_embed_from_grid(grid_w, embed_dim // 2)  # (B, N, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(grid_h, embed_dim // 2)  # (B, N, D/2)

    emb = torch.cat([emb_h, emb_w], dim=-1) # (B, L, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(pos, embed_dim):
    """
    embed_dim: output dimension for each position
    pos: a batch of list whose positions to be encoded: size (B, N)
    out: (B, N, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    out = torch.einsum('BL,D->BLD', pos, omega.to(pos)) # (B, N, D/2), outer product

    emb_sin = torch.sin(out) # (B, N, D/2)
    emb_cos = torch.cos(out) # (B, N, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=-1)  # (B, N, D)
    return emb

def get_3d_sincos_pos_embed_from_grid(grid, embed_dim, frac_coord_size=None, scale_ratio=1.0, time_dim=0):
    '''
    grid: (B, 3, N)
        N = H * W
        the first dimension represents width, and the second reprensents height
        e.g.,   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
    frac_coord_size: 
        if frac_coord_size != None: 
            fractional coordinates for positional embedding is used
        else: 
            absolute coordinates for positional embedding is used
    '''
    # assert embed_dim % 2 == 0
    if time_dim == 0:
        assert embed_dim % 3 == 0
        dim = embed_dim // 3
        time_dim = dim
    else:
        assert (embed_dim - time_dim) % 2 == 0
        dim = (embed_dim - time_dim) // 2
    
    grid = grid.float()
    if frac_coord_size != None:
        assert isinstance(frac_coord_size, (int, float))
        grid_w = grid[:, 0] / torch.max(grid[:, 0]) * frac_coord_size
        grid_h = grid[:, 1] / torch.max(grid[:, 1]) * frac_coord_size
        grid_t = grid[:, 2] / torch.max(grid[:, 2]) * frac_coord_size
    else:
        grid_w, grid_h, grid_t = grid[:, 0]*scale_ratio, grid[:, 1]*scale_ratio, grid[:, 2]*scale_ratio
    # use half of dimensions to encode grid_h
    emb_w = get_1d_sincos_pos_embed_from_grid(grid_w, dim)  # (B, N, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(grid_h, dim)  # (B, N, D/2)
    emb_t = get_1d_sincos_pos_embed_from_grid(grid_t, time_dim)  # (B, N, D/2)

    emb = torch.cat([emb_t, emb_h, emb_w], dim=-1) # (B, L, D)
    return emb

def get_21d_sincos_pos_embed_from_grid(grid, embed_dim, frac_coord_size=None, scale_ratio=1.0):
    '''
    grid: (B, 3, N)
        N = H * W
        the first dimension represents width, and the second reprensents height
        e.g.,   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
                [0. 1. 2. 3. 0. 1. 2. 3. 0. 1. 2. 3.]
                [0. 0. 0. 0. 1. 1. 1. 1. 2. 2. 2. 2.]
    frac_coord_size: 
        if frac_coord_size != None: 
            fractional coordinates for positional embedding is used
        else: 
            absolute coordinates for positional embedding is used
    '''
    assert embed_dim % 2 == 0
    dim = embed_dim // 2
    
    grid = grid.float()
    if frac_coord_size != None:
        assert isinstance(frac_coord_size, (int, float))
        grid_w = grid[:, 0] / torch.max(grid[:, 0]) * frac_coord_size
        grid_h = grid[:, 1] / torch.max(grid[:, 1]) * frac_coord_size
        grid_t = grid[:, 2] / torch.max(grid[:, 2]) * frac_coord_size
    else:
        grid_w, grid_h, grid_t = grid[:, 0]*scale_ratio, grid[:, 1]*scale_ratio, grid[:, 2]*scale_ratio
    # use half of dimensions to encode grid_h
    emb_w = get_1d_sincos_pos_embed_from_grid(grid_w, dim)
    emb_h = get_1d_sincos_pos_embed_from_grid(grid_h, dim)
    emb_t = get_1d_sincos_pos_embed_from_grid(grid_t, embed_dim)

    emb = torch.cat([emb_h, emb_w], dim=-1) + emb_t # (B, L, D)
    return emb

def get_time_sincos_pos_embed_from_grid(grid, embed_dim, frac_coord_size=None, scale_ratio=1.0):
    grid = grid.float()
    grid_t = grid[:, 0]*scale_ratio
    emb_t = get_1d_sincos_pos_embed_from_grid(grid_t, embed_dim)
    return emb_t


#################################################################################
#                                 interpolation                                 #
#################################################################################


def interpolate_sincos_pos_embed(embed_dim, ori_h, ori_w, tgt_h, tgt_w):
    from src.inf.models.dit import get_2d_sincos_pos_embed
    pos_embed = get_2d_sincos_pos_embed(embed_dim, ori_h, ori_w)
    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
    pos_embed = rearrange(pos_embed, '1 (h w) d -> 1 d h w', h=ori_h, w=ori_w)
    pos_embed = F.interpolate(pos_embed, (tgt_h, tgt_w), mode='bilinear')
    pos_embed = rearrange(pos_embed, '1 d h w -> 1 (h w) d')
    return pos_embed

def interpolate_sincos_pos_index(embed_dim, ori_h, ori_w, tgt_h, tgt_w):
    from src.inf.models.dit import get_2d_sincos_pos_embed_from_grid
    grid_h = np.arange(tgt_h, dtype=np.float32) * ori_h / tgt_h
    grid_w = np.arange(tgt_w, dtype=np.float32) * ori_w / tgt_w
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, tgt_h, tgt_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0)
    return pos_embed
