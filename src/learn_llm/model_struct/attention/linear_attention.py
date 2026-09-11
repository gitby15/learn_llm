import torch
import torch.nn as nn
import torch.nn.functional as F
EPS = 1e-6 # 用于防止0除

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
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        B, T, D = query.shape
        query = self.feature_map(q)
        key = self.feature_map(k)

        # 中间产物
        S = torch.zeros(B, T, D, D)
        Z = torch.zeros(B, T, D)

        output = torch.zeros(B, T, D)
        for t in range(N):
            q_t = q[:, :, t]
            k_t = k[:, :, t]
            v_t = v[:, :, t]

            kv_t = k_t.unsqueeze(-1) @ v_t.unsqueeze(-2)
            S = S + kv_t
            Z = Z + k_t

            numerator = q_t.unsqueeze(-2) @ S
            denominator = (q_t * Z).sum(dim=-1, keepdim=True)
            output[:, :, t] = numerator / (denominator.clamp_min(EPS))

        return output

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        return self.w_o(self._attention(query, key, value, mask))



if __name__ == "__main__":
    B, H, N, D = (2,3,4,5)
    Q = torch.ones(B, H, N, D)
    S = torch.zeros(B, H, D, D)
    for t in range(N):
        q = Q[:, :, t]
        S = S + q.unsqueeze(-1) @ q.unsqueeze(-2)

    print(S)

