import os
from datasets import Dataset
from transformers import AutoTokenizer
from learn_llm._utils_.pipeline import Pipeline, LoadNode, SaveNode, PipelineNode

_CWD_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_CWD_DIR, "data_outputs", "wikipedia")
MIN_CHUNK_LENGTH = 50


class TokenizeChunkNode(PipelineNode):
    name = "tokenize_chunk"

    def __init__(
        self,
        tokenizer,
        max_length: int,
        min_chunk: int = MIN_CHUNK_LENGTH,
        batch_size: int = 1024,
    ):
        self.tokenizer = tokenizer
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
                    chunk = token_ids[i : i + max_len]
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
        print(
            f"  排序完成，最短: {lengths[sorted_idx[0]]}, 最长: {lengths[sorted_idx[-1]]}"
        )
        return dataset


class PackDocumentsNode(PipelineNode):
    name = "pack"

    def __init__(self, tokenizer: AutoTokenizer, max_length: int):
        self.tokenizer = tokenizer
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






def get_tokenizer_dataset(take_len: int = None) -> Dataset:
    pipeline = Pipeline(
        [
            LoadNode(
                path="wikimedia/wikipedia",
                dataset_name="20231101.zh",
                split="train",
                check_dir='',
                take_len=take_len,
            ),
        ]
    )
    return pipeline.run()


def get_train_dataset(
        tokenizer: AutoTokenizer,
        context_max_len: int,
        data_take_len: int = None,
    ) -> Dataset:
    if os.path.exists(OUTPUT_DIR):
        print("预处理数据存在，直接加载...")
        dataset = Dataset.load_from_disk(OUTPUT_DIR)
        if data_take_len is not None:
            dataset = dataset.take(data_take_len)
        print(f"最终加载 {len(dataset)} 条数据")
        return dataset
    print("预处理数据不存在，开始数据处理流水线...")

    pipeline = Pipeline(
        [
            LoadNode(
                path="wikimedia/wikipedia",
                dataset_name="20231101.zh",
                split="train",
                check_dir=OUTPUT_DIR,
                take_len=data_take_len,
            ),
            TokenizeChunkNode(
                tokenizer=tokenizer,
                max_length=context_max_len
            ),
            SortByLengthNode(),
            PackDocumentsNode(
                tokenizer=tokenizer,
                max_length=context_max_len
            ),
            SaveNode(output_dir=OUTPUT_DIR),
        ]
    )
    return pipeline.run()
  


if __name__ == "__main__":
    a = get_train_dataset()
    # print(a)
