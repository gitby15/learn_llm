"""
数据预处理脚本：
1. 从 HuggingFace 加载 wikimedia/wikipedia 数据集
2. Tokenize 并按序列长度从小到大排序
3. 保存到 data_outputs/wikipedia 目录（datasets 可直读的格式）
"""

import os
from datasets import load_dataset, Dataset
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer

# 项目根目录，data_outputs 与 model_outputs 同级
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "data_outputs", "wikipedia")

MAX_LENGTH = 512





def pre_process():
    def tokenize_function(examples, tokenizer):
        texts = examples["text"]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
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
    tokenizer = MinimindTokenizer.get_tokenizer()

    print("正在加载数据集...")
    dataset = load_dataset(
        path="wikimedia/wikipedia",
        name="20231101.zh",
        split="train",
        streaming=False,  # 全量加载到内存，才能排序
    )
    print(f"数据集加载完成，共 {len(dataset)} 条")

    print("正在 tokenize...")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=1024,
        num_proc=os.cpu_count(),  # 多核 CPU 并行加速
        remove_columns=dataset.column_names,
    )
    print(f"Tokenize 完成，共 {len(tokenized)} 条")

    # 计算每条数据的有效 token 数（不含 padding），用于排序
    print("正在计算长度并排序...")
    lengths = [sum(mask) for mask in tokenized["attention_mask"]]
    # 按长度升序排序
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    tokenized = tokenized.select(sorted_indices)

    print(f"排序完成，最短: {lengths[sorted_indices[0]]} tokens, 最长: {lengths[sorted_indices[-1]]} tokens")

    print(f"正在保存到 {OUTPUT_DIR} ...")
    tokenized.save_to_disk(OUTPUT_DIR)
    print("保存完成！")


if __name__ == "__main__":
    main()