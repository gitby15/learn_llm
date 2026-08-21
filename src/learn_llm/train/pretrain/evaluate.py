import torch
import lm_eval
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
from learn_llm._utils_.model_path import ModelPath
from learn_llm.model.timllm import TimLLM
from learn_llm.model.model_config import TimLLMConfig
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer

# 注册自定义模型，使 AutoModel 能识别 timllm 类型
AutoConfig.register("timllm", TimLLMConfig)
AutoModelForCausalLM.register(TimLLMConfig, TimLLM)


def generate_samples(
    model: TimLLM,
    prompts: list[str] | None = None,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
):
    if prompts is None:
        prompts = [
            "今天天气真好，",
            "人工智能的发展",
            "在数学中，",
        ]

    tokenizer = MinimindTokenizer.get_tokenizer()
    model.eval()

    results = []
    for prompt in tqdm(prompts, desc="生成样本"):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "generated": generated_text})

    return results


def evaluate(
    model_dir: str,
    task: str = "hellaswag",
    batch_size: int = 128,
    device: str = "cuda",
):
    model_path = str(model_dir)
    print(f"从 {model_path} 加载模型")

    model = ModelPath.get_exist_model(TimLLM, model_dir)
    model = model.to(device)

    # 1. lm_eval 评测
    print("\n" + "=" * 50)
    print(f"{task} 评测...")
    print("=" * 50)
    results = lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, device=device, batch_size=batch_size),
        tasks=[task],
    )
    for task_name, task_metrics in results["results"].items():
        for metric, value in task_metrics.items():
            print(f"  {task_name}/{metric}: {value:.4f}" if isinstance(value, float) else f"  {task_name}/{metric}: {value}")

    # 2. 生成样本
    print("\n" + "=" * 50)
    print("生成样本...")
    print("=" * 50)
    samples = generate_samples(model)
    for s in samples:
        print(f"\n[Prompt]   {s['prompt']}")
        print(f"[Generated] {s['generated']}")

    return results, samples


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    evaluate(
        model_dir=ModelPath.PRETRAIN_SAVE,
        device=device,
    )

if __name__ == "__main__":
    main()