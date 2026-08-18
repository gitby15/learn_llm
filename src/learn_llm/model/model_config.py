from transformers import PreTrainedConfig


class TimLLMConfig(PreTrainedConfig):
    model_type = "timllm"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        gqa_head_dim: int = 64,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 512,
        rope_freq: int = 10000,
    ):
        # 所有自定义字段必须通过 kwargs 传给 PreTrainedConfig 注册
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            gqa_head_dim=gqa_head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_freq=rope_freq,
        )
        assert hidden_size == gqa_head_dim * num_attention_heads