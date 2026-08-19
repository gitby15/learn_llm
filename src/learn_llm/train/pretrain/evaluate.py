import math
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator
from learn_llm.model.timllm import TimLLM
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_llm.dataset.minimind import get_evaluate_dataset


def compute_perplexity(
    model: TimLLM,
    samples_skip: int = 0,
    samples_len: int = 500,
    batch_size: int = 32,
    max_eval_batches: int | None = None,
) -> dict:
    tokenizer = MinimindTokenizer().get_tokenizer()
    eval_dataset = get_evaluate_dataset(
        samples_skip=samples_skip,
        samples_len=samples_len,
        batch_size=batch_size,
        tokenizer=tokenizer,
    )

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            output = model(input_ids=input_ids, labels=labels)
            loss = output.loss

            valid_token_count = (labels != -100).sum().item()
            total_loss += loss.item() * valid_token_count
            total_tokens += valid_token_count

            if max_eval_batches is not None and batch_idx + 1 >= max_eval_batches:
                break

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = math.exp(avg_loss)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
    }


def generate_samples(
    model: TimLLM,
    prompts: list[str] | None = None,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_p: float = 0.9,
):
    if prompts is None:
        prompts = [
            "今天天气真好，",
            "人工智能的发展",
            "在数学中，",
        ]

    tokenizer = MinimindTokenizer().get_tokenizer()
    model.eval()

    results = []
    for prompt in prompts:
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
    model_dir: str = "./trained_model",
    samples_skip: int = 500000,
    samples_len: int = 500,
    batch_size: int = 32,
    max_eval_batches: int | None = None,
):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_default_device(DEVICE)
    print(f"using device: {DEVICE}")

    model = TimLLM.get_exist_model(DEVICE, model_dir)
    print(f"从 {model_dir} 加载模型成功")

    print("\n" + "=" * 50)
    print("计算 Perplexity...")
    print("=" * 50)
    metrics = compute_perplexity(
        model=model,
        samples_skip=samples_skip,
        samples_len=samples_len,
        batch_size=batch_size,
        max_eval_batches=max_eval_batches,
    )
    print(f"Loss:       {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"评估 Token 数: {metrics['total_tokens']}")

    print("\n" + "=" * 50)
    print("生成样本...")
    print("=" * 50)
    samples = generate_samples(model)
    for s in samples:
        print(f"\n[Prompt]   {s['prompt']}")
        print(f"[Generated] {s['generated']}")

    return metrics, samples


def main():
    evaluate(
        model_dir="./trained_model",
        samples_len=500,
        max_eval_batches=50,
    )    

if __name__ == "__main__":
    main()