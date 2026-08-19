import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from modelscope.hub.file_download import dataset_file_download
from transformers import AutoTokenizer

def _download_data():
    file_path = dataset_file_download(
        dataset_id='gongjy/minimind_dataset',
        file_path='pretrain_t2t_mini.jsonl'
    )
    print(f"Minimind Dataset 文件所在路径: {file_path}")
    return file_path

def tokenize_function(examples, tokenizer, max_length=512):
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = []
    for seq_ids, seq_mask in zip(input_ids, attention_mask):
        seq_labels = [tid if m == 1 else -100 for tid, m in zip(seq_ids, seq_mask)]
        labels.append(seq_labels)
    tokenized["labels"] = labels
    return tokenized


class StreamingIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterable_dataset, length=None):
        self.dataset = iterable_dataset
        self._length = length

    def __len__(self):
        if self._length is not None:
            return self._length
        raise TypeError(f"{type(self).__name__} has no length; pass `length` to constructor")

    def __iter__(self):
        yield from self.dataset


# 是一个很大的jsonl
def _get_stream_dataset(
        samples_skip:int,
        samples_len: int,
        batch_size: int,
        tokenizer: AutoTokenizer,
    ) -> StreamingIterableDataset:
    data_files = _download_data()
    dataset = load_dataset('json', data_files=data_files, split='train', streaming=True)
    dataset = dataset.skip(samples_skip)
    if samples_len >= 0:
        dataset = dataset.take(samples_len)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        batch_size=batch_size,
        remove_columns=["text"],
    )
    train_dataset = StreamingIterableDataset(tokenized_dataset, length=samples_len if samples_len >= 0 else None)
    return train_dataset

def get_train_dataset(samples_skip:int, samples_len: int, batch_size: int, tokenizer: AutoTokenizer) -> StreamingIterableDataset:
    return _get_stream_dataset(samples_skip, samples_len, batch_size, tokenizer)

def get_evaluate_dataset(samples_skip:int, samples_len: int, batch_size: int, tokenizer: AutoTokenizer) -> StreamingIterableDataset:
    return _get_stream_dataset(samples_skip, samples_len, batch_size, tokenizer)