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
_CWD_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_CWD_DIR, "data_outputs", "wikipedia")

MAX_LENGTH = 16384  # 与模型 max_position_embeddings 一致
MIN_CHUNK_LENGTH = 50  # 过滤太短的 chunk（几乎没有训练价值）


def pre_process():
    if os.path.exists(OUTPUT_DIR):
        print(f"数据集已存在: {OUTPUT_DIR}")
        print("跳过预处理")
        return

    def tokenize_and_chunk(examples, tokenizer):
        """tokenize 后将长文本切成多个 ≤MAX_LENGTH 的段，短文本保留原样。"""
        all_input_ids = []
        for text in examples["text"]:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(token_ids), MAX_LENGTH):
                chunk = token_ids[i:i + MAX_LENGTH]
                if len(chunk) >= MIN_CHUNK_LENGTH:
                    all_input_ids.append(chunk)
        return {"input_ids": all_input_ids}

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
        lambda x: tokenize_and_chunk(x, tokenizer),
        batched=True,
        batch_size=1024,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
    )
    print(f"Tokenize + 切块完成，共 {len(tokenized)} 条（原始 {len(dataset)} 篇）")

    # 计算每条数据的有效 token 数（不含 padding），用于排序
    print("正在计算长度并排序...")
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    # 按长度升序排序
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
    tokenized = tokenized.select(sorted_indices)

    print(f"排序完成，最短: {lengths[sorted_indices[0]]} tokens, 最长: {lengths[sorted_indices[-1]]} tokens")

    # 文档打包：将短文档拼接成满 MAX_LENGTH 的序列，用 EOS 隔开
    print("正在打包文档...")
    eos_id = tokenizer.eos_token_id
    packed_input_ids = []
    current_pack = []
    current_len = 0

    for ids in tokenized["input_ids"]:
        need = len(ids) + (1 if current_pack else 0)  # 非首篇需要 +1 放 EOS
        if current_len + need <= MAX_LENGTH:
            if current_pack:
                current_pack.append(eos_id)
            current_pack.extend(ids)
            current_len += need
        else:
            if current_pack:
                packed_input_ids.append(current_pack)
            current_pack = list(ids)
            current_len = len(ids)

    if current_pack:
        packed_input_ids.append(current_pack)

    tokenized = Dataset.from_dict({"input_ids": packed_input_ids})
    print(f"打包完成: {len(packed_input_ids)} 条，平均 {sum(len(p) for p in packed_input_ids) / len(packed_input_ids):.0f} tokens/条")

    print(f"正在保存到 {OUTPUT_DIR} ...")
    tokenized.save_to_disk(OUTPUT_DIR)
    print("保存完成！")

def get_train_dataset():
    if not os.path.exists(OUTPUT_DIR):
        print("预处理数据不存在，正在自动生成...")
        pre_process()
    return Dataset.load_from_disk(OUTPUT_DIR)

if __name__ == "__main__":

    log_path = os.path.join(_CWD_DIR, "logs", "test_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        def log(msg):
            print(msg)
            f.write(str(msg) + "\n")

        log("Test：开始加载数据")
        dataset = load_dataset(
            path="wikimedia/wikipedia",
            name="20231101.zh",
            split="train",
            streaming=True,
        )
        dataset.skip(100000)
        log("Test：加载数据完成")
        idx = 0
        for item in dataset:
            log(f"item: {idx} ===============")
            log(item)
            idx += 1
            if idx > 100:
                break
        log(f"日志已保存到 {log_path}")