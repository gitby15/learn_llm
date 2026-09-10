import os

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from learn_llm._utils_.model_path import ModelPath
from learn_llm.dataset.chinese_fineweb.sft import get_sft_train_dataset
from learn_llm.model.timllm import TimLLM
from learn_llm.model.timllm.model_config import TimLLMConfig


def _get_resume_checkpoint() -> str | None:
    if not os.path.isdir(ModelPath.SFT_CHECKPOINT):
        return None
    return get_last_checkpoint(ModelPath.SFT_CHECKPOINT)


def train():
    TimLLMConfig.register_for_auto_class("AutoConfig")
    TimLLM.register_for_auto_class("AutoModelForCausalLM")

    resume_checkpoint = _get_resume_checkpoint()
    model_path = resume_checkpoint or ModelPath.PRETRAIN_SAVE
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    print(
        f"{'继续 SFT checkpoint' if resume_checkpoint else '加载预训练模型'}: "
        f"{model_path}"
    )
    model = TimLLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    max_len = model.config.max_position_embeddings
    tokenizer.model_max_length = max_len

    dataset = get_sft_train_dataset(
        tokenizer=tokenizer,
        context_max_len=max_len,
        take_len=1000,
    )
    if len(dataset) < 2:
        raise ValueError("SFT 至少需要 2 条有效数据，才能划分训练集和验证集")

    eval_size = min(128, max(1, len(dataset) // 50))
    dataset_split = dataset.train_test_split(test_size=eval_size, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]

    supervised_tokens = sum(
        label != -100
        for labels in train_dataset["labels"]
        for label in labels
    )
    print(f"训练数据量: {len(train_dataset):,} 条")
    print(f"验证数据量: {len(eval_dataset):,} 条")
    print(f"训练集 assistant tokens: {supervised_tokens:,}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    training_args = TrainingArguments(
        output_dir=ModelPath.SFT_CHECKPOINT,
        save_total_limit=2,
        num_train_epochs=1,
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        prediction_loss_only=True,
        save_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=1.0,
        dataloader_num_workers=0,
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    try:
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存 checkpoint...")
        trainer._save_checkpoint(model, trial=None)
        return

    trainer.save_model(ModelPath.SFT_SAVE)
    tokenizer.save_pretrained(ModelPath.SFT_SAVE)


def main():
    train()


if __name__ == "__main__":
    main()
