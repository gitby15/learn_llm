from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from learn_llm.model.timllm import TimLLM
from learn_llm.model.model_config import TimLLMConfig
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_llm.dataset.minimind.sft import get_sft_train_dataset
from learn_llm._utils_.model_path import ModelPath
import torch


def train(
    samples_skip: int,
    samples_len: int,
    resume_dir: str,
):

    tokenizer = MinimindTokenizer.get_tokenizer()

    model = ModelPath.get_exist_model(TimLLM, resume_dir)
    if model is None:
        raise FileNotFoundError(f"从 {resume_dir} 加载预训练模型失败，返回空")

    train_dataset = get_sft_train_dataset(
        samples_skip=samples_skip,
        samples_len=samples_len,
        batch_size=1024,
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    # 3. SFT 训练参数（学习率比预训练低）
    training_args = TrainingArguments(
        output_dir=ModelPath.SFT_CHECKPOINT,
        save_total_limit=2,

        num_train_epochs=3,
        
        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        train_sampling_strategy="group_by_length", # 按长度分组采样，减少不必要的padding

        learning_rate=5e-5,
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 3},
        warmup_steps=100,

        logging_steps=20,
        save_steps=400,
        
        fp16=torch.cuda.is_available(),
        
        dataloader_num_workers=0,
        # 本地可以看训练进展
        report_to="tensorboard",
    )

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

    # 4. 保存 SFT 模型
    trainer.save_model(ModelPath.SFT_SAVE)
    tokenizer.save_pretrained(ModelPath.SFT_SAVE)


def main():
    train(samples_skip=0, samples_len=1000000, resume_dir=ModelPath.PRETRAIN_SAVE)


if __name__ == "__main__":
    main()