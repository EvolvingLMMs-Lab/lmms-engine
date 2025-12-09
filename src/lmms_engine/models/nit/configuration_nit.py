from typing import Optional

from transformers import PretrainedConfig


class NitConfig(PretrainedConfig):
    model_type = "nit"

    def __init__(
        self,
        input_size=32,
        patch_size=1,
        in_channels=32,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        encoder_depth=8,
        projector_dim=2048,
        z_dim=1280,
        use_checkpoint: bool = False,
        custom_freqs: str = "normal",
        theta: int = 10000,
        max_pe_len_h: Optional[int] = None,
        max_pe_len_w: Optional[int] = None,
        decouple: bool = False,
        ori_max_pe_len: Optional[int] = None,
        qk_norm: bool = False,
        **kwargs,
    ):
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
        self.qk_norm = qk_norm

        super().__init__(**kwargs)
