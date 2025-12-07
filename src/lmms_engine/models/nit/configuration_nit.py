from typing import Optional

from transformers import PretrainedConfig


class NitConfig(PretrainedConfig):
    model_type = "nit"

    def __init__(
        self,
        vae_name_or_path: str = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers",
        vae_dtype: str = "float32",
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
        custom_freqs: str = "normal",
        theta: int = 10000,
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
        compile: bool = False,
        **kwargs,
    ):
        self.vae_name_or_path = vae_name_or_path
        self.vae_dtype = vae_dtype
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.projector_dim = projector_dim
        self.z_dim = z_dim
        self.use_checkpoint = use_checkpoint
        self.custom_freqs = custom_freqs
        self.theta = theta
        self.max_pe_len_h = max_pe_len_h
        self.max_pe_len_w = max_pe_len_w
        self.decouple = decouple
        self.ori_max_pe_len = ori_max_pe_len
        self.compile = compile

        super().__init__(**kwargs)
