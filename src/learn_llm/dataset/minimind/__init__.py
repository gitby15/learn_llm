"""
数据预处理 Pipeline —— 仿 transformers.pipeline 风格，每个节点是独立可调用的 callable 对象。

节点列表:
    LoadNode          → 加载原始数据集
    TokenizeChunkNode → Tokenize + 长文切块
    SortByLengthNode  → 按长度排序
    PackDocumentsNode → 文档打包（短文档拼接填满序列）
    SaveNode          → 保存到磁盘

用法:
    pipeline = Pipeline([
        LoadNode(),
        TokenizeChunkNode(tokenizer),
        SortByLengthNode(),
        PackDocumentsNode(tokenizer, max_length=16384),
        SaveNode(output_dir),
    ])
    pipeline.run()
"""

import os
import time
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer

_CWD_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_CWD_DIR, "data_outputs", "wikipedia")
MAX_LENGTH = 16384
MIN_CHUNK_LENGTH = 50


# ---- 节点基类 ----

class PipelineNode(ABC):
    """仿 transformers.pipeline 的节点基类，每个节点是一个 callable"""
    name: str = "node"

    @abstractmethod
    def __call__(self, dataset: Dataset | None) -> Dataset:
        ...


# ---- 各节点实现 ----

class LoadNode(PipelineNode):
    name = "load"

    def __init__(self, dataset_path: str = "wikimedia/wikipedia",
                 dataset_name: str = "20231101.zh", split: str = "train"):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split

    def __call__(self, _dataset: None = None) -> Dataset:
        dataset = load_dataset(
            path=self.dataset_path,
            name=self.dataset_name,
            split=self.split,
            streaming=False,
        )
        print(f"  加载完成，共 {len(dataset)} 篇")
        return dataset


class TokenizeChunkNode(PipelineNode):
    name = "tokenize_chunk"

    def __init__(self, tokenizer=None, max_length: int = MAX_LENGTH,
                 min_chunk: int = MIN_CHUNK_LENGTH, batch_size: int = 1024):
        self.tokenizer = tokenizer or MinimindTokenizer.get_tokenizer()
        self.max_length = max_length
        self.min_chunk = min_chunk
        self.batch_size = batch_size

    def __call__(self, dataset: Dataset) -> Dataset:
        max_len, min_chunk = self.max_length, self.min_chunk

        def _fn(examples):
            all_ids = []
            for text in examples["text"]:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
                for i in range(0, len(token_ids), max_len):
                    chunk = token_ids[i:i + max_len]
                    if len(chunk) >= min_chunk:
                        all_ids.append(chunk)
            return {"input_ids": all_ids}

        tokenized = dataset.map(
            _fn,
            batched=True,
            batch_size=self.batch_size,
            num_proc=os.cpu_count(),
            remove_columns=dataset.column_names,
        )
        print(f"  Tokenize + 切块完成，共 {len(tokenized)} 条")
        return tokenized


class SortByLengthNode(PipelineNode):
    name = "sort"

    def __call__(self, dataset: Dataset) -> Dataset:
        lengths = [len(ids) for ids in dataset["input_ids"]]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        dataset = dataset.select(sorted_idx)
        print(f"  排序完成，最短: {lengths[sorted_idx[0]]}, 最长: {lengths[sorted_idx[-1]]}")
        return dataset


class PackDocumentsNode(PipelineNode):
    name = "pack"

    def __init__(self, tokenizer=None, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer or MinimindTokenizer.get_tokenizer()
        self.max_length = max_length

    def __call__(self, dataset: Dataset) -> Dataset:
        def _gen():
            eos = self.tokenizer.eos_token_id
            max_len = self.max_length
            current, current_len = [], 0

            for ids in dataset["input_ids"]:
                need = len(ids) + (1 if current else 0)
                if current_len + need <= max_len:
                    if current:
                        current.append(eos)
                    current.extend(ids)
                    current_len += need
                else:
                    if current:
                        yield {"input_ids": current}
                    current = list(ids)
                    current_len = len(ids)

            if current:
                yield {"input_ids": current}

        result = Dataset.from_generator(_gen)
        print(f"  打包完成: {len(result)} 条")
        return result


class SaveNode(PipelineNode):
    name = "save"

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = output_dir

    def __call__(self, dataset: Dataset) -> Dataset:
        dataset.save_to_disk(self.output_dir)
        return dataset


# ---- Pipeline 编排 ----

class Pipeline:
    """将多个 PipelineNode 串联执行。"""

    def __init__(self, nodes: list[PipelineNode]):
        self.nodes = nodes

    def run(self, skip_if_exists: bool = False):
        if skip_if_exists and os.path.exists(OUTPUT_DIR):
            print(f"[跳过] 数据已存在: {OUTPUT_DIR}")
            return
        dataset = None
        for node in self.nodes:
            t0 = time.time()
            print(f"\n[{node.name}] 开始...")
            dataset = node(dataset)
            print(f"[{node.name}] 完成，耗时 {time.time() - t0:.1f}s")
        print(f"\n流水线完成")


# ---- 默认流水线 ----

def _default_pipeline() -> Pipeline:
    tokenizer = MinimindTokenizer.get_tokenizer()
    return Pipeline([
        LoadNode(),
        TokenizeChunkNode(tokenizer),
        SortByLengthNode(),
        PackDocumentsNode(tokenizer),
        SaveNode(OUTPUT_DIR),
    ])


def get_train_dataset() -> Dataset:
    """加载预处理好的训练数据（不存在时自动生成）。"""
    if not os.path.exists(OUTPUT_DIR):
        print("预处理数据不存在，开始流水线...")
        _default_pipeline().run()
    return Dataset.load_from_disk(OUTPUT_DIR)


if __name__ == "__main__":
    a = get_train_dataset()
    # print(a)
