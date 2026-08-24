import os
from datasets import load_dataset, Dataset
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer
from learn_llm._utils_.pipeline import Pipeline, PipelineNode, PipelineAction

_CWD_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_CWD_DIR, "data_outputs", "minimind")
MAX_LENGTH = 1024
MIN_CHUNK_LENGTH = 50

class LoadNode(PipelineNode):
    name = "load"

    def __init__(
        self, path: str, data_files: str, split: str, check_dir: str
    ):
        self.path = path
        self.data_files = data_files
        self.split = split
        self.check_dir = check_dir

    def __call__(self, _dataset: None = None) -> Dataset:
        dataset = load_dataset(
            path=self.path,
            data_files=self.data_files,
            split=self.split,
            streaming=False,
        )
        
        dataset = dataset.take(300000)
        print(f"加载完成，共 {len(dataset)} 条数据")
        return dataset

    def _check(self) -> PipelineAction:
        if os.path.exists(self.check_dir):
            print(f"[跳过] 数据已存在: {self.check_dir}")
            return PipelineAction.INTERRUPT
        return PipelineAction.PASS


def _default_pipeline() -> Pipeline:
    tokenizer = MinimindTokenizer.get_tokenizer()
    return Pipeline(
        [
            LoadNode(
                path="jingyaogong/minimind_dataset",
                data_files="sft_t2t.jsonl",
                split="train",
                check_dir=OUTPUT_DIR,
            ),
        ]
    )


def get_train_dataset() -> Dataset:
    """加载预处理好的训练数据（不存在时自动生成）。"""
    if not os.path.exists(OUTPUT_DIR):
        print("预处理数据不存在，开始流水线...")
        _default_pipeline().run()
    return Dataset.load_from_disk(OUTPUT_DIR)


if __name__ == "__main__":
    print('minimind sft')
    load_node = LoadNode(
                    path="jingyaogong/minimind_dataset",
                    data_files="sft_t2t.jsonl",
                    split="train",
                    check_dir=OUTPUT_DIR,
                )
    dataset = load_node()
    print(f"长度: {len(dataset)}")
