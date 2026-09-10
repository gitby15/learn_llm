import torch
import torch.nn as nn
import torch.nn.functional as F


# 单头注意力，纯纯写来练习
class Attention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        # 单头注意力，dim就是全量数据
        self.w_q = nn.Linear(hidden_size, head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, head_dim, bias=False)

        self.w_o = nn.Linear(head_dim, hidden_size, bias=False)

    # 这里QKV的形状都是[B, T, C]
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # attention 公式
        attention = F.softmax(
            query @ key.transpose(-2, -1) / self.head_dim**0.05, dim=-1
        )
        result = attention @ value
        return self.w_o(result)


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
        return x.reshape(batch_size, -1, self.num_heads,  self.sub_head_dim).transpose(1, 2)
    
    def bhtd_to_btc(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return x.transpose(1,2).reshape(batch_size, -1, self.num_heads * self.sub_head_dim)

    # 把q,k,v从 [B, T, H*D] 转成 [B, H, T, D]
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        query = self.btc_to_bhtd(query)
        key = self.btc_to_bhtd(key)
        value = self.btc_to_bhtd(value)

        attention = F.softmax(
            query @ key.transpose(-2, -1) / self.sub_head_dim**0.05, dim=-1
        )

        result = self.bhtd_to_btc(attention @ value)
        result = self.w_o(result)
        return result


class GQA(nn.Module):
    def __init__(self, hidden_size: int, sub_head_dim: int, num_q_heads: int, num_kv_heads: int):
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
    
    # Todo: 我对torch的api还不够熟练，手写GQA有点烦，先让AI写了一版，回头再逐行学习
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size = q.size(0)
        q_len = q.size(1)

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # [B, Tq, Hq*D] -> [B, H_kv, G, Tq, D]
        query = query.reshape(
            batch_size,
            q_len,
            self.num_kv_heads,
            self.group_size,
            self.sub_head_dim,
        ).permute(0, 2, 3, 1, 4)

        # [B, Tkv, H_kv*D] -> [B, H_kv, Tkv, D]
        key = self.btc_to_bhtd(key, self.num_kv_heads)
        value = self.btc_to_bhtd(value, self.num_kv_heads)

        # [B, H_kv, G, Tq, Tkv]
        scores = torch.einsum(
            "bhgqd,bhkd->bhgqk",
            query,
            key,
        ) / self.sub_head_dim**0.5
        attention = F.softmax(scores, dim=-1)

        # [B, H_kv, G, Tq, D]
        result = torch.einsum(
            "bhgqk,bhkd->bhgqd",
            attention,
            value,
        )
        # [B, H_kv, G, Tq, D] -> [B, Tq, Hq*D]
        result = result.permute(0, 3, 1, 2, 4).reshape(
            batch_size,
            q_len,
            self.num_q_heads * self.sub_head_dim,
        )
        return self.w_o(result)
