from learn_llm._utils_.pipeline import Pipeline, PipelineNode
from datasets import Dataset, load_dataset


class LoadNode(PipelineNode):
    name = "load"

    def __init__(self, take_len: int = None):
        self.take_len = take_len
        pass

    def __call__(self, dataset=None) -> Dataset:
        dataset = load_dataset(
            path="opencsg/chinese-fineweb-edu-v2",
            split="train",
        )

        if self.take_len is not None:
            dataset = dataset.take(self.take_len)
        return dataset


def get_tokenizer_dataset(take_len: int = None) -> Dataset:
    pipeline = Pipeline(
        [
            LoadNode(take_len=take_len),
        ]
    )
    return pipeline.run()


if __name__ == "__main__":
    dataset = get_tokenizer_dataset()

    for item in dataset['text']:
        print(item)
        break
