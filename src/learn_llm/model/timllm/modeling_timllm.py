from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from learn_llm.model.components.attention import AttentionLayer
from learn_llm.model.timllm.model_config import TimLLMConfig


class TimLLM(PreTrainedModel, GenerationMixin):
    config_class = TimLLMConfig
    _tied_weights_keys = {"lm_head.weight": "embedding.weight"}

    def __init__(self, config: TimLLMConfig):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout_embed = nn.Dropout(config.dropout_rate)
        self.attention_layer = AttentionLayer(config)
        # LM Head: Linear(hidden -> vocab)，归一化由 AttentionLayer.final_norm 完成
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        start_pos = past_key_values.get_seq_length() if past_key_values is not None else 0

        # 1. Embedding: [B, T] -> [B, T, C]
        hidden_states = self.embedding(input_ids)
        hidden_states = self.dropout_embed(hidden_states)

        # 2. Transformer layers
        hidden_states, new_past_key_values = self.attention_layer(
            hidden_states, start_pos, past_key_values,
        )

        # 3. LM Head -> logits: [B, T, V]
        logits = self.lm_head(hidden_states)

        # 4. Loss（仅当提供 labels 时计算，且做正确的 shift）
        loss: torch.FloatTensor | None = None

        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]

            loss = cast(torch.FloatTensor, F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            ))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=new_past_key_values,
        )

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, **kwargs):
        super().tie_weights(**kwargs)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def get_size(self) -> tuple[int, dict[str, int]]:
        """返回 (总参数量, 各模块参数量)，自动去重 tied weights。"""
        seen = set()
        sub_module_size: dict[str, int] = {}
        for name, module in self.named_modules():
            count = 0
            for p in module.parameters(recurse=False):
                if p.data_ptr() not in seen:
                    seen.add(p.data_ptr())
                    count += p.numel()
            if count > 0:
                sub_module_size[name] = count
        return sum(sub_module_size.values()), sub_module_size
