import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from learn_llm._utils_.pipeline import Pipeline, PipelineAction, PipelineNode, SaveNode

_CWD_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_CWD_DIR, "data_outputs", "babylm_zho")
MIN_CHUNK_LENGTH = 50
DATASET_NAME = "BabyLM-community/babylm-zho"

class CheckExistNode(PipelineNode):
    name = "check_exist"
    def __call__(self, dataset=None) -> Dataset:
        pass
    def _check(self):
        if os.path.exists(OUTPUT_DIR):
            print(f"[跳过] 数据已存在: {OUTPUT_DIR}")
            return PipelineAction.INTERRUPT
        return PipelineAction.PASS

class LoadNode(PipelineNode):
    name = "load:" + DATASET_NAME

    def __init__(self, take_len: int):
        self.take_len = take_len
        pass

    def __call__(self, dataset=None) -> Dataset:
        dataset = load_dataset(
            DATASET_NAME,
            split='train', # 这个数据集只有train
        )
        if self.take_len is not None:
            dataset = dataset.take(self.take_len)
        return dataset


class TokenizeChunkNode(PipelineNode):
    name = "tokenize_chunk"
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        pass

    def __call__(self, dataset: Dataset):

        def _fn(examples):
            return {"input_ids": self.tokenizer(examples["text"], add_special_tokens=True)["input_ids"]}
        tokenized = dataset.map(
            _fn,
            batched=True,
            batch_size=64,
            num_proc=os.cpu_count(),
        )
        print(f"Tokenize 完成，共 {len(tokenized)} 条")
        return tokenized

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
            quota = max_len

            for ids in dataset['input_ids']:
                point = 0
                while True:
                    if quota <= 0: # quota用完了，就 yield 当前的 current
                        result = current
                        current = []
                        quota = max_len
                        yield {"input_ids": result}
                        break
                    if point >= len(ids) - 1: # ids 用完了，标记结束，并break到下一个ids
                        current.append(_eos)
                        quota -= 1
                        break

                    insert_len = min(quota, len(ids) - point - 1)
                    current.extend(ids[point:point + insert_len])
                    point += insert_len
                    quota -= insert_len

            if current:
                yield {"input_ids": current}

        result = Dataset.from_generator(_gen)
        
        print(f"  打包完成: {len(result)} 条, 平均每条长度: {sum(len(ids) for ids in result['input_ids']) /len(result):.2f}")
        return result

class SortNode(PipelineNode):
    name = "sort"
    def __call__(self, dataset: Dataset) -> Dataset:
        lengths = [len(ids) for ids in dataset["input_ids"]]
        sorted_idx = sorted(range(len(lengths)), key=lambda i: lengths[i])
        dataset = dataset.select(sorted_idx)
        print(
            f"  排序完成，最短: {lengths[sorted_idx[0]]}, 最长: {lengths[sorted_idx[-1]]}"
        )
        return dataset



    
def get_train_dataset(tokenizer: AutoTokenizer, context_max_len:int, data_take_len: int = None):
    pipeline = Pipeline(
        [
            CheckExistNode(),
            LoadNode(take_len=data_take_len),
            TokenizeChunkNode(tokenizer=tokenizer),
            # SortNode(),
            PackDocumentsNode(tokenizer=tokenizer, context_max_len=context_max_len),
            SaveNode(output_dir=OUTPUT_DIR)
        ]
    )
    pipeline.run()
    return Dataset.load_from_disk(OUTPUT_DIR)

def get_tokenizer_dataset(take_len: int = None) -> Dataset:
    pipeline = Pipeline(
        [
            LoadNode(take_len=take_len),
        ]
    )
    return pipeline.run()
