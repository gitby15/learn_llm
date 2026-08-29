import gc
import os
import time
from abc import ABC, abstractmethod
from datasets import Dataset, IterableDataset, load_from_disk
from transformers import AutoTokenizer
from enum import Enum


class PipelineAction(Enum):
    PASS = "pass"
    SKIP = "skip"
    INTERRUPT = "interrupt"


# ---- 节点基类 ----


class PipelineNode(ABC):
    name: str = "node"

    @abstractmethod
    def __call__(self, dataset: Dataset | None) -> Dataset: ...

    # 默认返回放行
    def _check(self) -> PipelineAction:
        return PipelineAction.PASS


# 数据pipeline一般都有这几个节点：
# 1. 从huggingface下载数据（早期简化学习，先从huggingface下载，不支持自己造）
# 2. 讲下载好的数据进行tokenize
# 3. 如果不做打包，就按照长度进行排序，如果要打包，就打包到统一长度
# 4. 保存到磁盘
class Pipeline:
    def __init__(self, nodes: list[PipelineNode]):
        self.nodes = nodes
    def run(self) -> Dataset:
        dataset = None
        try:
            for node in self.nodes:
                t0 = time.time()
                status = node._check()
                if status == PipelineAction.INTERRUPT:
                    dataset = node(dataset)
                    print(f"[{node.name}] 完成，提前中止流水线")
                    break
                elif status == PipelineAction.SKIP:
                    
                    print(f"跳过 [{node.name}] 节点")
                    continue
                else:
                    print(f"\n[{node.name}] 开始...")
                    dataset = node(dataset)
                print(f"[{node.name}] 完成，耗时 {time.time() - t0:.1f}s")
            print(f"\n流水线完成")
            return dataset
        finally:
            if isinstance(dataset, IterableDataset):
                del dataset
                gc.collect()

class CheckExistNode(PipelineNode):
    name = "check_exist"
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
    def __call__(self, dataset=None) -> Dataset:
        return load_from_disk(self.cache_dir)
    def _check(self):
        if os.path.exists(self.cache_dir):
            print(f"缓存目录已存在: {self.cache_dir}，直接使用缓存")
            return PipelineAction.INTERRUPT
        else:
            return PipelineAction.SKIP

class SaveNode(PipelineNode):
    name = "save"

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, dataset: Dataset) -> Dataset:
        dataset.save_to_disk(self.output_dir)
        return dataset


class PackDocumentsNode(PipelineNode):
    name = "pack"

    def __init__(self, tokenizer: AutoTokenizer, context_max_len: int):
        self.tokenizer = tokenizer
        self.context_max_len = context_max_len
        pass

    def __call__(self, dataset: Dataset) -> Dataset:
        def _gen():
            _eos = self.tokenizer.eos_token_id
            max_len = self.context_max_len
            current = []

            for ids in dataset['input_ids']:
                if not ids:
                    continue

                document = list(ids)
                if document[-1] != _eos:
                    document.append(_eos)

                point = 0
                while point < len(document):
                    quota = max_len - len(current)
                    insert_len = min(quota, len(document) - point)
                    current.extend(document[point:point + insert_len])
                    point += insert_len

                    if len(current) == max_len:
                        yield {"input_ids": current}
                        current = []

            if current:
                yield {"input_ids": current}

        result = Dataset.from_generator(_gen)
        
        print(f"  打包完成: {len(result)} 条, 平均每条长度: {sum(len(ids) for ids in result['input_ids']) /len(result):.2f}")
        return result
