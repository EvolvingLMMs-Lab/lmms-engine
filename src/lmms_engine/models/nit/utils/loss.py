import torch
import numpy as np
import torch.nn.functional as F

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))

class FlowMatchingLoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="lognormal",
            encoders=[], 
            accelerator=None, 
            latents_scale=None, 
            latents_bias=None,
            P_mean=0.0,
            P_std=1.0,
            sigma_data=1.0,
            unit_variance=False,
        ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.encoders = encoders
        self.accelerator = accelerator
        self.latents_scale = latents_scale
        self.latents_bias = latents_bias
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.unit_variance = unit_variance

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":
            alpha_t = torch.cos(t * torch.pi / 2)
            sigma_t = torch.sin(t * torch.pi / 2)
            d_alpha_t = -torch.pi / 2 * torch.sin(t * torch.pi / 2)
            d_sigma_t =  torch.pi / 2 * torch.cos(t * torch.pi / 2)
        elif self.path_type == 'triangle':
            alpha_t = torch.cos(t)
            sigma_t = torch.sin(t)
            d_alpha_t = -torch.sin(t)
            d_sigma_t =  torch.cos(t)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, batch_size, images, noises, model_kwargs=None, use_dir_loss=False, zs=[]):
        if model_kwargs == None:
            model_kwargs = {}
        # sample timestep according to log-normal distribution of sigmas following EDM
        rnd_normal = torch.randn((batch_size))
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        if self.path_type == "linear":      # [0, 1]
            t = sigma / (1 + sigma)        
        elif self.path_type == "cosine":    # [0, 1]
            t = 2 / np.pi * torch.atan(sigma)
        elif self.path_type == 'triangle':  # [0, pi/2]
            t = torch.atan(sigma / self.sigma_data)
        else:
            raise NotImplementedError
        t = t.to(device=images.device, dtype=images.dtype)
        
        time_input = t

        hw_list = model_kwargs['hw_list']
        seqlens = hw_list[:, 0] * hw_list[:, 1]
        t = torch.cat([t[i].unsqueeze(0).repeat(seqlens[i], 1, 1, 1) for i in range(batch_size)], dim=0)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t)
        
        if self.unit_variance:
            model_input = alpha_t * images / self.sigma_data + sigma_t * noises 
        else:
            model_input = alpha_t * images + sigma_t * noises
   
        if self.prediction == 'v':
            model_target = d_alpha_t * images + d_sigma_t * noises
        else:
            raise NotImplementedError() # TODO: add x or eps prediction
        
        model_kwargs['return_zs'] = True
        if self.unit_variance:
            model_output, zs_tilde = self.sigma_data * model(model_input, time_input, **model_kwargs)
        else:
            model_output, zs_tilde = model(model_input, time_input, **model_kwargs)
        
        denoising_loss = mean_flat((model_output - model_target) ** 2)
        denoising_loss = torch.nan_to_num(denoising_loss, nan=0, posinf=1e5, neginf=-1e5)
        loss = denoising_loss.mean()

        if use_dir_loss:
            directional_loss = mean_flat(1 - F.cosine_similarity(model_output, model_target, dim=1))
            directional_loss = torch.nan_to_num(directional_loss, nan=0, posinf=1e5, neginf=-1e5)
            loss += directional_loss.mean()
        
        proj_loss = 0.
        if zs != [] and zs != None:
            for i, (z, z_tilde) in enumerate(zip(zs, zs_tilde)):
                proj_loss += 1 - torch.cosine_similarity(z, z_tilde, dim=-1).mean()
            proj_loss = torch.nan_to_num(proj_loss, nan=0, posinf=1e5, neginf=-1e5)

        return loss, proj_loss
