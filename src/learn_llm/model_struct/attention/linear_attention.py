# 线性注意力的核心是 S = S(t-1) + Kt*Vt
# 相关变种：
# RetNet： S = γ*S(t-1) + Kt*Vt, 0<γ<1
# Gate Linear Attention: S = Gate*S(t-1) + Kt*Vt
# RWKV、SSM、Mamba 等结构后面再继续学习

import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-6  # 用于防止0除

class LinearAttention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        self.w_q = nn.Linear(hidden_size, head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, head_dim, bias=False)
        self.w_o = nn.Linear(head_dim, hidden_size, bias=False)

    # 核函数
    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        # 尝试把softmax(QKt) 替换成f(Q)*f(Kt)*V, 目的是先去算f(Kt)*V
        return F.elu(x) + 1.0

    def _attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        B, T, D = q.shape
        q = self.feature_map(q)
        k = self.feature_map(k)

        if past_key_value is not None:
            S, Z = past_key_value
        else:
            # 中间产物
            S = torch.zeros(B, D, D, device=q.device, dtype=q.dtype)
            # 用于归一化的分母
            Z = torch.zeros(B, D, device=q.device, dtype=q.dtype)

        output = torch.zeros(B, T, D, device=q.device, dtype=q.dtype)
        for t in range(T):
            # 形状是(B, D)
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]

            if attention_mask is not None:
                # 线性注意力的mask核心是不要污染S和Z，所以可以盯着S和Z来实现
                mask = attention_mask[:, t].unsqueeze(-1)
                # Q也可以不乘以mask，因为计算出来的output，在后面的流程中不会被用到，要么是decode出来的值被抛弃，要么是训练时计算loss的时候被抛弃
                q_t = q_t * mask
                k_t = k_t * mask
                # 实际上v可以不乘mask，因为在计算kv_t的时候，相同位置会跟k的0相乘，结果还是0
                v_t = v_t * mask

            # [B, D, 1] * [B, 1, D] = [B, D, D]
            kv_t = k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)
            S = S + kv_t
            Z = Z + k_t

            # [B , 1, D] @ [B, D, D] = [B, 1, D]
            numerator = q_t.unsqueeze(-2) @ S
            # [B, D] * [B, D] sum -> [B, 1] -> [B, 1, 1]
            denominator = (q_t * Z).sum(dim=-1, keepdim=True).unsqueeze(-1)
            # [B, 1, D] / [B, 1, 1] -> [B, 1, D]
            attention = numerator / (denominator.clamp_min(EPS))
            output[:, t, :] = attention.squeeze(-2)

        return output, (S, Z)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        # 形状是[B, T]
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        attention, kv_cache = self._attention(
            query, key, value, attention_mask, past_key_value
        )

        return self.w_o(attention), kv_cache


if __name__ == "__main__":
    B, D, C = (2, 3, 4)
    q = torch.ones(B, D, C)
    z = torch.ones(B, D, 1)

    result = q / z

    print(result.shape)
    print(result)
