from datasets import load_dataset
from modelscope.hub.file_download import dataset_file_download
from transformers import AutoTokenizer
from learn_llm.dataset.minimind import StreamingIterableDataset

def _download_data():
    data_files = dataset_file_download(
        dataset_id='gongjy/minimind_dataset',
        file_path='sft_t2t_mini.jsonl'
    )
    print(f"SFT Dataset 文件所在路径: {data_files}")
    dataset = load_dataset('json', data_files=data_files, split='train', streaming=True)
    return dataset

# 暂时不处理reasoning_content
def _tokenize_single_conversation(conversations, tokenizer, max_length=512):
    input_ids = []
    labels = []

    bos = tokenizer.bos_token          # e.g. "<|im_start|>"
    eos = tokenizer.eos_token          # e.g. "<|im_end|>"

    for turn in conversations:
        role = turn["role"]
        content = turn["content"]

        if role == "system":
            segment = f"{bos}system\n{content}{eos}\n"
            seg_ids = tokenizer.encode(segment, add_special_tokens=False)
            input_ids.extend(seg_ids)
            # 计算loss的时候跳过提问部分
            labels.extend([-100] * len(seg_ids))

        elif role == "user":
            segment = f"{bos}user\n{content}{eos}\n"
            seg_ids = tokenizer.encode(segment, add_special_tokens=False)
            input_ids.extend(seg_ids)
            # 计算loss的时候跳过提问部分
            labels.extend([-100] * len(seg_ids))

        elif role == "assistant":
            header = f"{bos}assistant\n"
            
            footer = f"{eos}\n"
            header_ids = tokenizer.encode(header, add_special_tokens=False)
            content_ids = tokenizer.encode(content, add_special_tokens=False)
            footer_ids = tokenizer.encode(footer, add_special_tokens=False)

            seg_ids = header_ids + content_ids + footer_ids
            seg_labels = [-100] * len(header_ids) + content_ids + [-100] * len(footer_ids)

            input_ids.extend(seg_ids)
            labels.extend(seg_labels)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def _tokenize_sft_conversation(examples, tokenizer, max_length=16384):
    """batch 版本的 tokenize，处理一批 conversations。"""
    all_input_ids = []
    all_labels = []

    for conversations in examples["conversations"]:
        result = _tokenize_single_conversation(conversations, tokenizer, max_length)
        all_input_ids.append(result["input_ids"])
        all_labels.append(result["labels"])

    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
    }


def _get_sft_stream_dataset(
    samples_skip: int,
    samples_len: int,
    batch_size: int,
    tokenizer: AutoTokenizer,
) -> StreamingIterableDataset:
    dataset = _download_data()
    dataset = dataset.skip(samples_skip)
    if samples_len >= 0:
        dataset = dataset.take(samples_len)

    tokenized_dataset = dataset.map(
        lambda x: _tokenize_sft_conversation(x, tokenizer),
        batched=True,
        batch_size=batch_size,
    )
    train_dataset = StreamingIterableDataset(
        tokenized_dataset,
        length=samples_len if samples_len >= 0 else None,
    )
    return train_dataset


def get_sft_train_dataset(
    samples_skip: int,
    samples_len: int,
    batch_size: int,
    tokenizer: AutoTokenizer,
) -> StreamingIterableDataset:
    return _get_sft_stream_dataset(samples_skip, samples_len, batch_size, tokenizer)


def get_sft_evaluate_dataset(
    samples_skip: int,
    samples_len: int,
    batch_size: int,
    tokenizer: AutoTokenizer,
) -> StreamingIterableDataset:
    return _get_sft_stream_dataset(samples_skip, samples_len, batch_size, tokenizer)


if __name__ == "__main__":
    from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer
    tokenizer = MinimindTokenizer.get_tokenizer()
    dataset = get_sft_evaluate_dataset(
        samples_skip=0 ,
        samples_len=5,
        batch_size=2,
        tokenizer=tokenizer,
    )
    for i, sample in enumerate(dataset):
        print(f"=== 第 {i + 1} 条 ===", sample)
