from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class SiTConfig(PretrainedConfig):
    model_type = "sit"

    def __init__(
        self,
        # SiT model parameters
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        # VAE parameters
        vae_path="stabilityai/sd-vae-ft-ema",
        # Create Transport parameters
        path_type="Linear",
        prediction="velocity",
        loss_weight=None,
        train_eps=None,
        sample_eps=None,
        cfg_scale=1.0,
        **kwargs,
    ):
        # SiT model parameters
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.class_dropout_prob = class_dropout_prob
        self.num_classes = num_classes
        self.learn_sigma = learn_sigma
        self.cfg_scale = cfg_scale

        self.image_size = input_size * 8

        self.model_params = {
            "input_size": input_size,
            "patch_size": patch_size,
            "in_channels": in_channels,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "class_dropout_prob": class_dropout_prob,
            "num_classes": num_classes,
            "learn_sigma": learn_sigma,
        }

        # VAE parameters
        self.vae_path = vae_path

        # Create Transport parameters
        self.path_type = path_type
        self.prediction = prediction
        self.loss_weight = loss_weight
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        super().__init__(**kwargs)
