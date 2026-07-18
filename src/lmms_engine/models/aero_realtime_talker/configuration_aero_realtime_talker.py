# src/lmms_engine/models/aero_realtime_talker/configuration_aero_realtime_talker.py
"""AeroRealtime talker configs (ported from Qwen3-TTS).

Two-level codebook talker mirroring Qwen3-Omni / Qwen3-TTS:
- The trunk (``AeroRealtimeTalkerConfig``) autoregressively predicts codec
  group 0 across frames.
- The dense ``code_predictor`` (``AeroRealtimeTalkerCodePredictorConfig``)
  predicts residual groups 1..num_code_groups-1 within each frame.

``AeroRealtimeTalkerConfig`` nests ``AeroRealtimeTalkerCodePredictorConfig``
via ``sub_configs``.
"""

from transformers.configuration_utils import PretrainedConfig


class AeroRealtimeTalkerCodePredictorConfig(PretrainedConfig):
    model_type = "aero_realtime_talker_code_predictor"

    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 5,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        sliding_window: int | None = None,
        num_code_groups: int = 16,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 0,
        layer_types: list | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.num_code_groups = num_code_groups
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        # The code predictor is fully bidirectional/dense (no sliding window):
        # every layer is full_attention. ``AeroRealtimeTalkerCodePredictorAttention`` /
        # decoder layer index ``config.layer_types[layer_idx]``.
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        self.layer_types = layer_types
        super().__init__(pad_token_id=pad_token_id, **kwargs)


class AeroRealtimeTalkerConfig(PretrainedConfig):
    model_type = "aero_realtime_talker"

    sub_configs = {"code_predictor_config": AeroRealtimeTalkerCodePredictorConfig}

    def __init__(
        self,
        vocab_size: int = 3072,
        hidden_size: int = 1024,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_theta: float = 1000000.0,
        rope_scaling: dict | None = None,
        sliding_window: int | None = None,
        num_code_groups: int = 16,
        text_hidden_size: int = 2048,
        text_vocab_size: int = 151936,
        # codec stream special tokens (Qwen3-TTS-12Hz, confirmed)
        codec_bos_id: int = 2149,
        codec_eos_id: int = 2150,
        codec_pad_id: int = 2148,
        codec_nothink_id: int = 2155,
        speaker_id: dict | None = None,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 0,
        tie_word_embeddings: bool = False,
        code_predictor_config=None,
        **kwargs,
    ):
        if code_predictor_config is None:
            code_predictor_config = AeroRealtimeTalkerCodePredictorConfig()
        elif isinstance(code_predictor_config, dict):
            code_predictor_config = AeroRealtimeTalkerCodePredictorConfig(**code_predictor_config)
        self.code_predictor_config = code_predictor_config

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_theta = rope_theta
        # The trunk attention applies multimodal (mrope) rope and reads
        # ``rope_scaling["mrope_section"]`` + ``rope_scaling["interleaved"]``.
        # mrope_section must sum to head_dim // 2 (== 128 // 2 == 64). The
        # default below is the real Qwen3-TTS-0.6B value ([24, 20, 20]).
        if rope_scaling is None:
            rope_scaling = {"mrope_section": [24, 20, 20], "interleaved": True, "rope_type": "default"}
        self.rope_scaling = rope_scaling
        self.sliding_window = sliding_window
        self.num_code_groups = num_code_groups
        self.text_hidden_size = text_hidden_size
        self.text_vocab_size = text_vocab_size
        self.codec_bos_id = codec_bos_id
        self.codec_eos_id = codec_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_nothink_id = codec_nothink_id
        self.speaker_id = speaker_id if speaker_id is not None else {"ryan": 3061}
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
