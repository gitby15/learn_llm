from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# 单头注意力，纯纯写来练习
class NormalAttention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        # 单头注意力，dim就是全量数据
        self.w_q = nn.Linear(hidden_size, head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, head_dim, bias=False)

        self.w_o = nn.Linear(head_dim, hidden_size, bias=False)

    # 这里最简单，直接套公式就好了
    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # 这个环节可能会是一个显存尖刺，它的空间复杂度是O(B * Tq * Tk)
        logits = query @ key.transpose(-2, -1)  # Q * K^t
        logits = logits / self.head_dim**0.5  # Q * K^t / sqrt(D)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        score = F.softmax(logits, dim=-1)
        return score @ value


    # kv_cache的空间复杂度：2 * D * TPast
    def _kv_cache_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,  # decode的时候，Mask是空的
        past_key_value: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if past_key_value is not None:
            # kv_cache 的形状是[B, Tpast, D], k和v的形状是[B, Tnew. D]
            # mask的形状是[Tq, Tpast + Tnew]
            key_cache, value_cache = past_key_value
            key = torch.cat([key_cache, key], dim=1)
            value = torch.cat([value_cache, value], dim=1)
        attention = self._attention(query, key, value, mask)
        return attention, (key, value)

    # 这里QKV的形状都是[B, T, D]
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        past_key_value: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        attention, kv_cache = self._kv_cache_attention(
            query, key, value, mask, past_key_value
        )
        return self.w_o(attention), kv_cache


class MHA(nn.Module):

    def __init__(self, hidden_size: int, sub_head_dim: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.sub_head_dim = sub_head_dim
        self.num_heads = num_heads

        self.w_q = nn.Linear(hidden_size, num_heads * sub_head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, num_heads * sub_head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, num_heads * sub_head_dim, bias=False)
        self.w_o = nn.Linear(num_heads * sub_head_dim, hidden_size, bias=False)

    def btc_to_bhtd(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.reshape(batch_size, -1, self.num_heads, self.sub_head_dim).transpose(
            1, 2
        )

    def bhtd_to_btc(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.sub_head_dim
        )

    # 将D维度，拆成不同的头，做并行计算，然后再合并
    def _attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
    ):
        query = self.btc_to_bhtd(query)
        key = self.btc_to_bhtd(key)
        value = self.btc_to_bhtd(value)
        logits = query @ key.transpose(-2, -1)
        logits = logits / self.sub_head_dim**0.5
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        score = F.softmax(logits, dim=-1)
        result = score @ value
        result = self.bhtd_to_btc(result)
        return result

    # 其实kv_cache的视线还是很简单的，现在还是学习项目，暂时不考虑性能优化，这些算子都不会被使用，真正会用到的算子，需要考虑出高性能版本
    def _kv_cache_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        past_key_value: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if past_key_value is not None:
            key_cache, value_cache = past_key_value
            key = torch.cat([key_cache, key], dim=1)
            value = torch.cat([value_cache, value], dim=1)
        attention = self._attention(query, key, value, mask)
        return attention, (key, value)

    # 把q,k,v从 [B, T, H*D] 转成 [B, H, T, D]
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        past_key_value: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        attention, kv_cache = self._kv_cache_attention(
            query, key, value, mask, past_key_value
        )
        result = self.w_o(attention)
        return result, kv_cache


class GQA(nn.Module):
    def __init__(
        self, hidden_size: int, sub_head_dim: int, num_q_heads: int, num_kv_heads: int
    ):
        super().__init__()
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        self.group_size = num_q_heads // num_kv_heads
        self.hidden_size = hidden_size
        self.sub_head_dim = sub_head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.w_q = nn.Linear(hidden_size, num_q_heads * sub_head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, num_kv_heads * sub_head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, num_kv_heads * sub_head_dim, bias=False)
        self.w_o = nn.Linear(num_q_heads * sub_head_dim, hidden_size, bias=False)

    def btc_to_bhtd(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        batch_size = x.size(0)
        return x.reshape(batch_size, -1, num_heads, self.sub_head_dim).transpose(1, 2)

    def _attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
    ):
        batch_size = q.size(0)
        q_len = q.size(1) # 在kv cache的场景，query可能只有1个Token，但是k和v仍然是满编的
        kv_len = k.size(1)
        group_size = self.group_size
        # [B, T, Hkv*G*D] -> [B, T, Hkv, G, D] -> [B, G, Hkv, T, D]
        grouped_query = q.reshape(
            batch_size, q_len, self.num_kv_heads, group_size, self.sub_head_dim
        ).transpose(1, 3)
        # [B, T, Hkv*D] -> [B, T, Hkv, D] -> [B, Hkv, T, D]
        key = k.reshape(
            batch_size, kv_len, self.num_kv_heads, self.sub_head_dim
        ).transpose(1, 2)
        value = v.reshape(
            batch_size, kv_len, self.num_kv_heads, self.sub_head_dim
        ).transpose(1, 2)

        # t表示q_len，s表示kv_len
        logits = torch.einsum(
            "bgktd,bkds->bgkts",
            grouped_query,
            key.transpose(-2, -1),
        )
        logits = logits / self.sub_head_dim**0.5
        if mask is not None:
            # Todo: 需要留意一下mask的形状，现在logits是[B, G, Hkv, Tq, Tkv], mask需要匹配这个形状
            logits = logits.masked_fill(mask == 0, float("-inf"))
        score = F.softmax(logits, dim=-1)
        
        result = torch.einsum(
            "bgkts,bksd->bgktd",
            score,
            value,
        )

        # [B, G, H_kv, T, D] -> [B, T, H_kv, G, D] -> [B, T, Hkv*G*D]
        result = result.transpose(1, 3).reshape(
            batch_size, q_len, self.num_kv_heads * self.group_size * self.sub_head_dim
        )
        return result

    def _kv_cache_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None,
        past_key_value: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        if past_key_value is not None:
            key_cache, value_cache = past_key_value
            key = torch.cat([key_cache, key], dim=1)
            value = torch.cat([value_cache, value], dim=1)
        attention = self._attention(query, key, value, mask)
        return attention, (key, value)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None,
        past_key_value: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        result, kv_cache = self._kv_cache_attention(query, key, value, mask, past_key_value)
        return self.w_o(result), kv_cache


class FastGQA(nn.Module):
    """使用融合投影与 PyTorch SDPA/FlashAttention 的高性能自注意力。"""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if num_q_heads % num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads")
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0, 1)")

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.dropout_p = dropout_p
        self.q_size = num_q_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        # 自注意力中 Q/K/V 输入相同，合并为一次 GEMM。
        self.w_qkv = nn.Linear(hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.w_o = nn.Linear(self.q_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        query, key, value = self.w_qkv(x).split(
            (self.q_size, self.kv_size, self.kv_size), dim=-1
        )

        query = query.view(
            batch_size, seq_len, self.num_q_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=self.num_q_heads != self.num_kv_heads,
        )
        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, self.q_size)
        return self.w_o(attention)
