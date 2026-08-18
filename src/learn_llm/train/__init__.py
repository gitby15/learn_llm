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

# Todo: 弄清楚为啥会遇到这个accelerate的兼容性问题
class SimpleTensorDataset(torch.utils.data.Dataset):
    """将 HuggingFace Dataset 转为简单 TensorDataset，绕过 accelerate 兼容问题"""
    def __init__(self, hf_dataset):
        self.input_ids = torch.tensor([item["input_ids"] for item in hf_dataset], dtype=torch.long)
        self.attention_mask = torch.tensor([item["attention_mask"] for item in hf_dataset], dtype=torch.long)
        self.labels = torch.tensor([item["labels"] for item in hf_dataset], dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }



# samples_len -1表示不限制数据量，即使用所有数据样本
def train(
        samples_start_index:int = 0,
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
    if samples_len != -1:
        dataset = dataset.select(range(samples_start_index, samples_start_index + samples_len))

    # 3. 预处理：tokenize
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )

    # 3.1 转为简单 TensorDataset
    train_dataset = SimpleTensorDataset(tokenized_dataset)

    # 4. 数据整理器
    data_collator = default_data_collator

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        per_device_train_batch_size=128,
        gradient_accumulation_steps=4,
        num_train_epochs=1, # 对于LLM来说，1~3 epoch就够了，需要弄清楚原理，我大概理解是需要一定的泛化能力
        learning_rate=3e-4,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=100,
        logging_steps=50,          # 每步都输出 loss，在进度条中显示
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),  # 有 GPU 时开启混合精度
        dataloader_pin_memory=False,      # default_device 已设 cuda，tensor 已在 GPU 上
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
    train(samples_start_index=300000,samples_len=700000)
    # train(100, resume=True)  # 继续训练：从 ./trained_model 加载权重，换新数据

if __name__ == "__main__":
    train(samples_len=100)