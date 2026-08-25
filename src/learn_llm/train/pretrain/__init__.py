from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from learn_llm._utils_.model_path import ModelPath
from learn_llm.model.timllm import TimLLM
from learn_llm.model.timllm.model_config import TimLLMConfig
from learn_llm.tokenizer.babylm_zho import get_tokenizer
# from learn_llm.dataset.babylm.babylm_zho import get_train_dataset
from learn_llm.dataset.wikipedia import get_train_dataset

def train(resume_dir: str):
    TimLLMConfig.register_for_auto_class("AutoConfig")
    TimLLM.register_for_auto_class("AutoModelForCausalLM")

    tokenizer = get_tokenizer()
    model = ModelPath.get_exist_model(TimLLM, resume_dir)
    if model is None:
        print("从头开始训练")
        config = TimLLMConfig(
            vocab_size=tokenizer.vocab_size,
        )
        model = TimLLM(config)

    train_dataset = get_train_dataset(tokenizer = tokenizer, context_max_len=model.config.max_position_embeddings)
    tokenizer.model_max_length = model.config.max_position_embeddings  # 关键：限制 collator 的 chunk 长度
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=ModelPath.PRETRAIN_CHECKPOINT,
        save_total_limit=2,

        num_train_epochs=2, # 训练轮数

        auto_find_batch_size=True,
        # per_device_train_batch_size=64,
        # gradient_accumulation_steps=4,
        # train_sampling_strategy="group_by_length", # 按长度分组采样，减少不必要的padding

        # 学习率相关的参数
        learning_rate=1e-4,
        # 用三角函数，学习率会平滑一些
        lr_scheduler_type="cosine",
        # lr_scheduler_kwargs={"num_cycles": 5},
        warmup_ratio=0.05,

        logging_steps=30,
        save_steps=400,

        # 有 GPU 时开启混合精度
        fp16=True,  # DEBUG: 暂时关闭 fp16 排查 loss 异常
        report_to="tensorboard",
        dataloader_num_workers=8,
        torch_compile=True,  # DEBUG: 暂时关闭 compile 排查 loss 异常
    )

    print("开始训练：")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存当前模型...")

    trainer.save_model(ModelPath.PRETRAIN_SAVE)
    tokenizer.save_pretrained(ModelPath.PRETRAIN_SAVE)


def main():
    train(resume_dir=ModelPath.PRETRAIN_SAVE)

if __name__ == "__main__":
    main()
