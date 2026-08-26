import torch

from learn_llm.model.timllm.model_config import TimLLMConfig

__all__ = ["TimLLM", "TimLLMConfig"]


def __getattr__(name: str):
    if name == "TimLLM":
        from learn_llm.model.timllm.modeling_timllm import TimLLM
        return TimLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def calculate_model_size():
    from learn_llm.model.timllm.modeling_timllm import TimLLM
    from learn_llm.tokenizer.babylm_zho import get_tokenizer
    tokenize = get_tokenizer()
    config = TimLLMConfig(vocab_size=tokenize.vocab_size)
    model = TimLLM(config)
    total_size, sub_module_size = model.get_size()
    for name, size in sub_module_size.items():
        print(f"-{name}: {size / 1e6} MB")
    print("===================================")
    print(f"模型参数量: {total_size / 1e6:.2f}M")

def test():
    from learn_llm.model.timllm.modeling_timllm import TimLLM
    config = TimLLMConfig(vocab_size=100)
    timllm = TimLLM(config)
    input = torch.randint(1, 10, (1, 10), dtype=torch.long)
    label = torch.ones(1, 10, dtype=torch.long)
    output = timllm(input_ids=input, labels=label)
    print(output)



if __name__ == "__main__":
    calculate_model_size()
   
