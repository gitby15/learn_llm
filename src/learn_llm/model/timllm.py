from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from learn_llm.model.model_config import TimLLMConfig
from learn_llm.model.components.attention import AttentionLayer

class TimLLM(PreTrainedModel, GenerationMixin):
    config_class = TimLLMConfig

    def __init__(self, config: TimLLMConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout_embed = nn.Dropout(config.dropout_rate)
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
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        **kwargs, # Todo: 研究一下transformer框架都会传什么东西进来
    ) -> CausalLMOutputWithPast:

        # 1. Embedding: [B, T] -> [B, T, C]
        hidden_states = self.embedding(input_ids)
        hidden_states = self.dropout_embed(hidden_states)

        # 2. Transformer layers
        hidden_states = self.attention_layer(hidden_states)

        # 3. LM Head -> logits: [B, T, V]
        logits = self.lm_head(hidden_states)

        # 4. Loss（仅当提供 labels 时计算，且做正确的 shift）
        loss: torch.FloatTensor | None = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
    
            loss = cast(torch.FloatTensor, F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
    @staticmethod
    def get_exist_model(device: torch.device, resume_dir: str = "./trained_model"):
        try:
            return TimLLM.from_pretrained(resume_dir).to(device)
        except:
            print(f"从 {resume_dir} 加载模型失败，返回空")
            return None



def test():
    from learn_llm.model.model_config import TimLLMConfig
    config = TimLLMConfig(vocab_size=100)
    timllm = TimLLM(config)
    input = torch.randint(1, 10, (1, 10), dtype=torch.long)
    label = torch.ones(1, 10, dtype=torch.long)
    output = timllm(input_ids=input, labels=label)
    print(output)

if __name__ == "__main__":
    test()
