import os

from transformers import AutoTokenizer

from learn_llm._utils_.pipeline import (
    PackDocumentsNode,
    Pipeline,
    PipelineNode,
    SaveNode,
    CheckExistNode,
)
from datasets import Dataset, load_dataset

_CWD_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_CWD_DIR, "data_outputs", "chinese_fineweb_edu_v2")
DATASET_NAME = "opencsg/chinese-fineweb-edu-v2"


class LoadNode(PipelineNode):
    name = "load"

    def __init__(self, take_len: int = None):
        self.take_len = take_len
        pass

    def __call__(self, dataset=None) -> Dataset:
        # 这个数据集非常非常大，几百G，必须流式加载
        dataset = load_dataset(
            path=DATASET_NAME,
            split="train",
            streaming=True,
        )

        if self.take_len is not None:
            # 流式打乱顺序
            dataset = dataset.shuffle(buffer_size=50000, seed=43).take(self.take_len)
        return dataset


class TokenizeChunkNode(PipelineNode):
    name = "tokenize_chunk"

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        pass

    def __call__(self, dataset: Dataset):
        def _fn(examples):
            return {
                "input_ids": self.tokenizer(examples["text"], add_special_tokens=True)[
                    "input_ids"
                ]
            }

        tokenized = dataset.map(
            _fn,
            batched=True,
        )
        return tokenized


def get_train_dataset(
    tokenizer: AutoTokenizer, context_max_len: int, take_len: int = None
) -> Dataset:
    pipeline = Pipeline(
        [
            CheckExistNode(cache_dir=OUTPUT_DIR),
            LoadNode(take_len=take_len),
            TokenizeChunkNode(tokenizer=tokenizer),
            PackDocumentsNode(tokenizer=tokenizer, context_max_len=context_max_len),
            SaveNode(output_dir=OUTPUT_DIR),
        ]
    )
    dataset = pipeline.run()
    total_tokens = len(dataset) * context_max_len
    print(
        f"预训练数据总量: {len(dataset):,} 条 × {context_max_len} = {total_tokens:,} tokens ({total_tokens / 1e9:.2f}B)"
    )
    return dataset


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(_CWD_DIR, "model_outputs", "pretrain", "save"))
    context_max_len = 2048
    dataset = get_train_dataset(
        tokenizer=tokenizer, context_max_len=context_max_len, take_len=50
    )

    for idx, item in enumerate(dataset["text"]):
        print(f"idx: {idx}, len: {len(item)}, item: {item}")
