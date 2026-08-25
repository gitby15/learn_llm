import torch
import torch.nn as nn
import torch.nn.functional as F
from learn_llm.model.timllm.model_config import TimLLMConfig

class Rope(nn.Module):
    def __init__(self, config: TimLLMConfig):
        super().__init__()
        self.dim = config.gqa_head_dim
        self.base = config.rope_freq

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.inv_freq: torch.Tensor  # 消除 register_buffer 产生的 Tensor | Module 类型歧义

    def forward(self, x):
        return self.rope_embedding(x)

    def rope_embedding(self, x):
        seq_len = x.size(-2)
        t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        freqs = torch.outer(t, self.inv_freq.to(x.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        x_rotated = torch.cat((-x[..., self.dim // 2:], x[..., :self.dim // 2]), dim=-1)
        return x * cos + x_rotated * sin


class GQAAttention(nn.Module):
    def __init__(self, config: TimLLMConfig, rope: Rope):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.gqa_head_dim

        self.w_q = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.rope = rope


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # q, k, v的形状：[B, T, C]
        b = q.size(0)
        t_q = q.size(1)
        t_kv = k.size(1)

        query = self.w_q(q)
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        query = query.reshape(b, t_q, self.num_attention_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        query = self.rope(query)
        

        key = self.w_k(k)
        # [B, T, C] -> [B, T, H, D] -> [B, H, T, D]
        key = key.reshape(b, t_kv, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key = self.k_norm(key)
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


class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络：gate_proj + up_proj + down_proj"""
    def __init__(self, config: TimLLMConfig):
        super().__init__()
        expanded_size = int(config.hidden_size * 8 / 3)
        self.gate_proj = nn.Linear(config.hidden_size, expanded_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, expanded_size, bias=False)
        self.down_proj = nn.Linear(expanded_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-Norm 结构：残差 + LN + Attention/FFN"""
    def __init__(self, config: TimLLMConfig, rope: Rope):
        super().__init__()
        self.layer_norm_1 = nn.RMSNorm(config.hidden_size)
        self.attention = GQAAttention(config, rope)
        self.dropout_attn = nn.Dropout(config.dropout_rate)
        self.layer_norm_2 = nn.RMSNorm(config.hidden_size)
        self.ffn = SwiGLUFFN(config)
        self.dropout_ffn = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.layer_norm_1(x)
        attn_out = self.attention(attn_in, attn_in, attn_in)
        attn_out = x + self.dropout_attn(attn_out)

        ffn_in = self.layer_norm_2(attn_out)
        ffn_out = self.ffn(ffn_in)
        result = attn_out + self.dropout_ffn(ffn_out)

        return result





class AttentionLayer(nn.Module):
    def __init__(self, config: TimLLMConfig):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(config, Rope(config))
            for _ in range(config.num_hidden_layers)
        ])
        self.final_norm = nn.RMSNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = x
        for block in self.blocks:
            attention_output = block(attention_output)

        return self.final_norm(attention_output)


if __name__ == "__main__":
    from learn_llm.model.timllm.model_config import TimLLMConfig
    config = TimLLMConfig(vocab_size=100000)
    rope = Rope(config)
    print(rope)