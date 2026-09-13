import torch
import torch.nn.functional as F
from torch import nn
from learn_llm.model_struct.ops.activations import SwiGluFFn


class SimpleMoeFFN(nn.Module):
    def __init__(self, hidden_size: int, up_size: int, num_experts: int, topk: int):
        super().__init__()
        assert topk <= num_experts
        assert topk > 0
        self.topk = topk
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGluFFn(hidden_size, up_size) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, C]
        original_shape = x.shape
        hidden_size = x.size(-1)
        # [B * T, C] 把B和T拉平，后面好计算一点
        x_flat = x.reshape(-1, hidden_size)
        # [B * T, num_experts] 针对每个token，计算专家分数
        # 这句话的语义，是给每个token，计算每个专家的分数，后面会选最高的k个专家
        router_logits = self.gate(x_flat)

        # routing_weights[B*T, topk]：为每个token选出来的topk专家分数，从高到低排序
        # selected_experts[B*T, topk]：为每个token选出来的topk专家索引（因为排过序了，所以用索引表示是哪个专家）
        routing_weights, selected_experts = torch.topk(
            router_logits, dim=-1, k=self.topk
        )
        # 做softmax，这样所有专家的比例加起来等于1
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(
            x.dtype
        )

        # [B * T, C] 先初始化一个0，再依次把专家的输出加权求和
        output_flat = torch.zeros_like(x_flat)

        for expert_idx, expert in enumerate(self.experts):
            # 找出有哪些token选择了这个专家，以及对应专家在routing_weights的索引
            token_idx, weight_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue  # 一个客户都没有，就跳过这个专家

            # x_flat[token_idx] 形状是[N, C]， N是token_idx的元素个数
            expert_output = expert(x_flat[token_idx])
            expert_output = expert_output * routing_weights[
                token_idx, weight_idx
            ].unsqueeze(-1)
            output_flat.index_add_(0, token_idx, expert_output)
        output = output_flat.reshape(original_shape)
        return output



if __name__ == "__main__":
    from learn_llm.utils.seed import set_seed
    from learn_llm.utils.device import get_cached_device
    set_seed(42)
    
    a = torch.randn(2,4,6, device=get_cached_device())
    b = a[1,3,5]
    print(a)
    print(b)
    print(torch.where(a == b))
    