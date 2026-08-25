from transformers import PreTrainedConfig


class TimLLMConfig(PreTrainedConfig):
    model_type = "timllm"

    def __init__(
        self,
        vocab_size: int = 6480,
        num_hidden_layers: int = 8,
        gqa_head_dim: int = 128,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 2048,
        rope_freq: int = 10000,
        dropout_rate: float = 0.05,
        tie_word_embeddings=True,
        **kwargs,
    ):
        kwargs.pop("hidden_size", None)
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=gqa_head_dim * num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            gqa_head_dim=gqa_head_dim,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_freq=rope_freq,
            dropout_rate=dropout_rate,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        