from learn_llm._utils_.pipeline import Pipeline, PipelineNode
from datasets import Dataset, IterableDataset, load_dataset

DATASET_NAME = "opencsg/chinese-fineweb-edu-v2"
FALLBACK_TOTAL_LENGTH = 1_500_000
SHUFFLE_BUFFER_SIZE = 1_000

class LoadNode(PipelineNode):
    name = "load"

    def __init__(self, take_len: int = None):
        self.take_len = take_len
        pass

    @staticmethod
    def _get_total_length(dataset: IterableDataset) -> int:
        splits = dataset.info.splits
        split_info = splits.get("train") if splits is not None else None
        total_length = getattr(split_info, "num_examples", None)
        if isinstance(total_length, int) and total_length > 0:
            print(f"从数据集元信息读取到总量: {total_length:,}")
            return total_length

        print(f"元信息未提供总量，使用兜底值: {FALLBACK_TOTAL_LENGTH:,}")
        return FALLBACK_TOTAL_LENGTH

    def __call__(self, dataset=None) -> IterableDataset:
        dataset = load_dataset(
            path=DATASET_NAME,
            split="train",
            streaming=True,
        )

        if self.take_len is not None:
            if self.take_len <= 0:
                raise ValueError("take_len 必须大于 0")

            total_length = self._get_total_length(dataset)
            sample_count = min(self.take_len, total_length)

            # shuffle 会先打乱数据文件顺序；较小缓冲区只承担局部随机，
            # 避免在内存中同时保留大量长文档。
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, seed=42)

            def _select_uniformly(_, index):
                current_bucket = index * sample_count // total_length
                next_bucket = (index + 1) * sample_count // total_length
                return next_bucket > current_bucket

            dataset = dataset.filter(_select_uniformly, with_indices=True).take(sample_count)
            print(
                f"均匀抽样: 目标 {sample_count:,} 条，"
                f"平均每 {total_length / sample_count:.2f} 条选取 1 条"
            )
        return dataset


def get_tokenizer_dataset(take_len: int) -> Dataset:
    pipeline = Pipeline(
        [
            LoadNode(take_len=take_len),
        ]
    )
    return pipeline.run()


if __name__ == "__main__":
    dataset = get_tokenizer_dataset(take_len=500)

    for item in dataset['text']:
        print(item)
        print("--- item len ---", len(item))
        break
