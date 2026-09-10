import os
import asyncio
from learn_llm.tokenizers.qwen35_06b import get_tokenizer
from datasets import Dataset, load_dataset

_CWD_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_CWD_DIR, "data_outputs", "Fineweb-Edu-Chinese-V2.1")
DATASET_NAME = "opencsg/Fineweb-Edu-Chinese-V2.1"
# 计划训练的模型大小是100M左右
# 根据经验，高性价比数据比例是20倍，我们的模型小，所以多训练一些，我们做30倍，大概3BToken
# 这个数据集很大，但是4-5分的部分，大概是4.5B Token，我们就只用这部分最优质的数据来做训练



tokenize_queue = asyncio.Queue()

# 流式加载，一边加载一边送入消息队列
async def load():
    dataset = load_dataset(
        path=DATASET_NAME,
        split="train",
        data_files="4_5/*.parquet",
        streaming=True,
    )
    return dataset
    
async def tokenize(dataset):
    tokenizer = get_tokenizer()
    def _fn(examples):
        return {
            "input_ids": tokenizer(examples["text"], add_special_tokens=True)[
                "input_ids"
            ]
        }
    tokenized_dataset = dataset.map(
        _fn,
        batched=True,
    )
    return tokenized_dataset

async def pack(dataset):
        tokenizer = get_tokenizer()

        def _gen():
            _eos = tokenizer.eos_token_id
            max_len = 8192 # Todo: 从模型参数里面读
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

        packed_dataset = Dataset.from_generator(_gen)
        return packed_dataset

async def save(dataset):
    dataset.save_to_disk(OUTPUT_DIR)

if __name__ == "__main__":
    async def test():
        dataset = await load()
        tokenized_dataset = await tokenize(dataset)
        packed_dataset = await pack(tokenized_dataset)
        await save(packed_dataset)

    asyncio.run(test())




