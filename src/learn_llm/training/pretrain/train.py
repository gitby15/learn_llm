from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from learn_llm._utils_.model_path import ModelPath
from learn_llm.model.timllm import TimLLM
from learn_llm.model.timllm.model_config import TimLLMConfig
from learn_llm.tokenizer.chinese_fineweb.train import get_tokenizer
from learn_llm.dataset.chinese_fineweb.pretrain import get_train_dataset
# from learn_llm.tokenizer.babylm_zho.train import get_tokenizer
# from learn_llm.dataset.babylm.babylm_zho import get_train_dataset as get_babylm_data
# from learn_llm.dataset.wikipedia import get_train_dataset as get_wiki_data


def train():
    TimLLMConfig.register_for_auto_class("AutoConfig")
    TimLLM.register_for_auto_class("AutoModelForCausalLM")

    tokenizer = get_tokenizer()

    model = ModelPath.get_latest_checkpoint_model(TimLLM, ModelPath.PRETRAIN_CHECKPOINT)
    is_resume = True
    if model is None:
        print("从头开始训练")
        is_resume = False
        config = TimLLMConfig(vocab_size=tokenizer.vocab_size)
        model = TimLLM(config)

    max_len = model.config.max_position_embeddings
    tokenizer.model_max_length = max_len

    dataset = get_train_dataset(tokenizer=tokenizer, context_max_len=max_len, take_len=1500000)
    dataset_split = dataset.train_test_split(test_size=1024, seed=42)
    train_dataset = dataset_split["train"]
    eval_dataset = dataset_split["test"]
    train_tokens = len(train_dataset) * max_len
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"训练数据量：{len(train_dataset):,} 条")
    print(f"训练 token 数：{train_tokens:,} ({train_tokens / 1e9:.2f}B)")
    print(f"验证数据量：{len(eval_dataset):,} 条，{len(eval_dataset) * max_len:,} tokens")
    print(
        f"可训练参数量：{trainable_parameters:,}，"
        f"训练 token/参数比：{train_tokens / trainable_parameters:.2f}"
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=ModelPath.PRETRAIN_CHECKPOINT,
        save_total_limit=5,
        num_train_epochs=1,
        auto_find_batch_size=True,
        train_sampling_strategy="sequential",
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        logging_steps=30,
        eval_strategy="steps",
        eval_steps=2000,
        prediction_loss_only=True,
        save_steps=2000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_grad_norm=1.0,
        report_to="tensorboard",
        dataloader_num_workers=8,
        torch_compile=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    try:
        trainer.train(resume_from_checkpoint=is_resume)
    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存 checkpoint...")
        trainer._save_checkpoint(model, trial=None)
        return

    trainer.save_model(ModelPath.PRETRAIN_SAVE)
    tokenizer.save_pretrained(ModelPath.PRETRAIN_SAVE)


def main():
    train()

if __name__ == "__main__":
    main()
