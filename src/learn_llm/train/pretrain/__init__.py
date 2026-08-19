from transformers import Trainer, TrainingArguments, default_data_collator
from learn_llm.model.timllm import TimLLM
from learn_llm.model.model_config import TimLLMConfig
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_llm.dataset.minimind import get_train_dataset
import torch

# samples_len -1表示不限制数据量，即使用所有数据样本
def train(
        samples_skip:int = 0,
        samples_len: int = 100,
        resume_dir: str = "./trained_model"
    ):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_default_device(DEVICE)
    print(f"using device: {DEVICE}")

    tokenizer = MinimindTokenizer().get_tokenizer()
    model = TimLLM.get_exist_model(DEVICE, resume_dir)
    if model is None:
        print("从头开始训练")
        config = TimLLMConfig(
            vocab_size=tokenizer.vocab_size,
        )
        model = TimLLM(config)
    else:
        print(f"从 {resume_dir} 加载模型继续训练") 

    # 2. 加载数据集
    per_device_train_batch_size=128
    train_dataset = get_train_dataset(
        samples_skip=samples_skip,
        samples_len=samples_len,
        batch_size=per_device_train_batch_size,
        tokenizer=tokenizer,
    )
    # Todo: 弄清楚这个是干啥的
    data_collator = default_data_collator

    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=4,
        # 开了流式数据集，现在只支持跑一轮训练
        num_train_epochs=1,

        # 学习率相关的参数
        learning_rate=6e-4,
        # 用三角函数，学习率会平滑一些
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=100,
        # 每步都输出 loss，在进度条中显示
        logging_steps=20,
        save_steps=500,
        save_total_limit=2,
        # 有 GPU 时开启混合精度
        fp16=torch.cuda.is_available(),
        # default_device 已设 cuda，tensor 已在 GPU 上
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


    # 8. 保存模型
    trainer.save_model("./trained_model")
    tokenizer.save_pretrained("./trained_model")


def main():
    train(samples_skip=0,samples_len=1000000)
    # train(100, resume=True)  # 继续训练：从 ./trained_model 加载权重，换新数据

if __name__ == "__main__":
    main()