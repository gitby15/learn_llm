from transformers import Trainer, TrainingArguments, default_data_collator
from learn_llm.model.timllm import TimLLM
from learn_llm.model.model_config import TimLLMConfig
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_llm.dataset.minimind import get_train_dataset
import torch



def tokenize_function(examples, tokenizer, max_length=512):
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = []
    for seq_ids, seq_mask in zip(input_ids, attention_mask):
        seq_labels = [tid if m == 1 else -100 for tid, m in zip(seq_ids, seq_mask)]
        labels.append(seq_labels)
    tokenized["labels"] = labels
    return tokenized

class StreamingIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable_dataset, length=None):
        self.dataset = iterable_dataset
        self._length = length

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError(f"{type(self).__name__} has no length; pass `length` to constructor")

    def __iter__(self):
        yield from self.dataset



# samples_len -1表示不限制数据量，即使用所有数据样本
def train(
        sample_skip:int = 0,
        samples_len: int = 100,
        resume_dir: str = "./trained_model"
    ):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch.set_default_device(DEVICE)
    print(f"using device: {DEVICE}")

    # 1. 加载 tokenizer和模型
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
    dataset = get_train_dataset()

    
    
    # 2.1 限制数据量（只取前 samples_len 条）
    dataset = dataset.skip(sample_skip)
    if samples_len >= 0:
        dataset = dataset.take(samples_len)

    # 3. 预处理：流式 tokenize（这个map在真正被加载的时候才会执行）
    per_device_train_batch_size=128
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=per_device_train_batch_size,
        remove_columns=["text"],
    )

    train_dataset = StreamingIterableDataset(tokenized_dataset, length=samples_len if samples_len >= 0 else None)

    # 4. 数据整理器
    data_collator = default_data_collator

    # 5. 训练参数
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
        logging_steps=50,
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

    # 7. 训练
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
    train(sample_skip=300000,samples_len=300000)
    # train(100, resume=True)  # 继续训练：从 ./trained_model 加载权重，换新数据

if __name__ == "__main__":
    train(samples_len=100)