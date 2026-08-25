import os
import time
from abc import ABC, abstractmethod
from datasets import load_dataset, Dataset
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
        for node in self.nodes:
            t0 = time.time()
            status = node._check()
            if status == PipelineAction.INTERRUPT:
                print(f"流水线提前结束于[{node.name}]")
                break
            elif status == PipelineAction.SKIP:
                print(f"[{node.name}] 跳过")
                continue
            else:
                print(f"\n[{node.name}] 开始...")
                dataset = node(dataset)
            print(f"[{node.name}] 完成，耗时 {time.time() - t0:.1f}s")
        print(f"\n流水线完成")
        return dataset


class LoadNode(PipelineNode):
    name = "load"

    def __init__(
        self, path: str, dataset_name: str, split: str, check_dir: str, take_len: int = None
    ):
        self.path = path
        self.dataset_name = dataset_name
        self.split = split
        self.check_dir = check_dir
        self.take_len = take_len

    def __call__(self, _dataset: None = None) -> Dataset:
        dataset = load_dataset(
            path=self.path,
            name=self.dataset_name,
            split=self.split,
            streaming=False,
        )
        if self.take_len is not None:
            dataset = dataset.take(self.take_len)
        print(f"加载完成，共 {len(dataset)} 条数据")
        return dataset

    def _check(self) -> PipelineAction:
        if os.path.exists(self.check_dir):
            print(f"[跳过] 数据已存在: {self.check_dir}")
            return PipelineAction.INTERRUPT
        return PipelineAction.PASS

class SaveNode(PipelineNode):
    name = "save"

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, dataset: Dataset) -> Dataset:
        dataset.save_to_disk(self.output_dir)
        return dataset