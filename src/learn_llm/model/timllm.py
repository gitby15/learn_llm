import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from learn_llm.model.model_config import TimLLMConfig
from learn_llm.model.components.attention import AttentionLayer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.set_default_device(DEVICE)
print(f"using device: {DEVICE}")

class TimLLM(PreTrainedModel):
    config_class = TimLLMConfig

    def __init__(self, config: TimLLMConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.attention_layer = AttentionLayer(config)
        # LM Head: hidden -> vocab logits
        self.lm_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size * 2, bias=False),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.vocab_size, bias=False),
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: torch.LongTensor = None,
    ) -> CausalLMOutputWithPast:

        # 1. Embedding: [B, T] -> [B, T, C]
        attention_input = self.embedding(input_ids)

        # 2. Transformer layers
        hidden_states = self.attention_layer(attention_input)

        # 3. LM Head -> logits: [B, T, V]
        logits = self.lm_head(hidden_states)

        # 4. Loss（仅当提供 labels 时计算，且做正确的 shift）
        loss = None

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )


def test():
    from learn_llm.model.model_config import TimLLMConfig
    config = TimLLMConfig()
    timllm = TimLLM(config)
    input = torch.randint(1, 10, (1, 10), dtype=torch.long)
    label = torch.ones(1, 10, dtype=torch.long)
    output = timllm(input, label)
    print(output)

if __name__ == "__main__":
    test()
