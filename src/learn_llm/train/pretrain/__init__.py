from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from learn_llm._utils_.model_path import ModelPath
from learn_llm.model.timllm import TimLLM
from learn_llm.model.model_config import TimLLMConfig
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_llm.dataset.minimind import get_train_dataset
import torch

def train(resume_dir: str):

    tokenizer = MinimindTokenizer.get_tokenizer()
    model = ModelPath.get_exist_model(TimLLM, resume_dir)
    if model is None:
        print("从头开始训练")
        config = TimLLMConfig(
            vocab_size=tokenizer.vocab_size,
        )
        model = TimLLM(config)

    # Batch 128在执行的过程中，大概会吃掉18GB的显存
    
    train_dataset = get_train_dataset()
    # Todo: 弄清楚这个是干啥的
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        
        output_dir=ModelPath.PRETRAIN_CHECKPOINT,
        save_total_limit=2,

        num_train_epochs=3, # 训练轮数

        auto_find_batch_size=True,
        gradient_accumulation_steps=4,
        # train_sampling_strategy="group_by_length", # 按长度分组采样，减少不必要的padding

        # 学习率相关的参数
        learning_rate=5e-4,
        # 用三角函数，学习率会平滑一些
        lr_scheduler_type="cosine_with_restarts",
        lr_scheduler_kwargs={"num_cycles": 5},
        warmup_steps=100,

        logging_steps=30,
        save_steps=400,

        # 有 GPU 时开启混合精度
        fp16=torch.cuda.is_available(),
        report_to="tensorboard",
        dataloader_num_workers=4,
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


    trainer.save_model(ModelPath.PRETRAIN_SAVE)
    tokenizer.save_pretrained(ModelPath.PRETRAIN_SAVE)


def main():
    train(resume_dir=ModelPath.PRETRAIN_SAVE)

if __name__ == "__main__":
    main()