# 激活函数的算子
# 手写一个SwiGlu和Silu
import argparse
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F



def silu(x: torch.Tensor) -> torch.Tensor:
    return x * F.sigmoid(x)

@torch.compile()
class SwiGluFFn(nn.Module):
    def __init__(self, hidden_size: int, up_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, up_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, up_size, bias=False)
        self.down_proj = nn.Linear(up_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        gate = silu(gate)
        up = self.up_proj(x)
        hidden = gate * up
        result = self.down_proj(hidden)
        return result
