from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from learn_llm.model.timllm import TimLLM
from learn_llm.model.model_config import TimLLMConfig
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_llm.dataset.minimind.sft import get_sft_train_dataset
from learn_llm._utils_.model_path import ModelPath
import torch


def train(
    samples_skip: int = 0,
    samples_len: int = 100,
):

    tokenizer = MinimindTokenizer.get_tokenizer()

    model = ModelPath.get_exist_model(TimLLM, ModelPath.PRETRAIN_SAVE)
    if model is None:
        raise FileNotFoundError(f"从 {ModelPath.PRETRAIN_SAVE} 加载模型失败，返回空")

    # 2. 加载 SFT 数据集
    per_device_train_batch_size = 32
    train_dataset = get_sft_train_dataset(
        samples_skip=samples_skip,
        samples_len=samples_len,
        batch_size=per_device_train_batch_size,
        tokenizer=tokenizer,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

    # 3. SFT 训练参数（学习率比预训练低）
    training_args = TrainingArguments(
        output_dir=ModelPath.SFT_CHECKPOINT,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=3,

        learning_rate=5e-5,
        lr_scheduler_type="cosine",

        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to="none",
        remove_unused_columns=False,
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
    train(samples_skip=0, samples_len=1000000)


if __name__ == "__main__":
    main()