import torch
import torch.nn as nn
import torch.nn.functional as F
from learn_llm.model.model_config import TimLLMConfig

class Rope(nn.Module):
    def __init__(self, config: TimLLMConfig):
        super().__init__()
        self.dim = config.gqa_head_dim
        self.max_seq_len = config.max_position_embeddings
        self.base = config.rope_freq

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        t = torch.arange(self.max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x):
        return self.ropa_embedding(x)

    def ropa_embedding(self, x):
        seq_len = x.size(-2)
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        x_rotated = torch.cat((-x[..., self.dim // 2:], x[..., :self.dim // 2]), dim=-1)
        return x * cos + x_rotated * sin


class GQAAttention(nn.Module):
    def __init__(self, config: TimLLMConfig, rope: Rope):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.gqa_head_dim
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads

        self.w_q = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = rope


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # q, k, v的形状：[B, H, T, D]
        b = q.size(0)
        t_q = q.size(1)
        t_kv = k.size(1)

        query = self.w_q(q)
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        query = query.reshape(b, t_q, self.num_attention_heads, self.head_dim).transpose(1, 2)
        query = self.rope(query)
        

        key = self.w_k(k)
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        key = key.reshape(b, t_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key = self.rope(key)

        value = self.w_v(v)
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        value = value.reshape(b, t_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        attention_out = F.scaled_dot_product_attention(
            query, key, value,
            is_causal=True,
            enable_gqa=True,
        )

        # [B, H, T, D] -> [B, T, H, D] -> [B, T, C]
        attention_out = attention_out.transpose(1, 2).reshape(b, t_q, self.num_attention_heads * self.head_dim)

        return self.w_o(attention_out)


class TransformerBlock(nn.Module):
    """Pre-Norm 结构：残差 + LN + Attention/FFN"""
    def __init__(self, config: TimLLMConfig, rope: Rope):
        super().__init__()
        self.config = config
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.attention = GQAAttention(config, rope)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.layer_norm_1(x)
        
        attention_output = self.attention(normed, normed, normed)

        # 残差连接
        attention_output = x + attention_output
        ffn_input = self.layer_norm_2(attention_output)
        ffn_output = ffn_input + self.feed_forward(ffn_input)
        return ffn_output


class AttentionLayer(nn.Module):
    def __init__(self, config: TimLLMConfig):
        super().__init__()
        self.rope = Rope(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config, self.rope)
            for _ in range(config.num_hidden_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.num_attention_heads = config.num_attention_heads
        self.gqa_head_dim = config.gqa_head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        attention_output = x
        for block in self.blocks:
            attention_output = block(attention_output)

        return self.final_norm(attention_output)


if __name__ == "__main__":
    from learn_llm.model.model_config import TimLLMConfig
    config = TimLLMConfig()
    rope = Rope(config)
    print(rope)