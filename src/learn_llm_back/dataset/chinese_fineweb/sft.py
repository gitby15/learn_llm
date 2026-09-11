import os
from collections.abc import Iterator
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset
from transformers import PreTrainedTokenizerBase

from learn_llm._utils_.pipeline import (
    CheckExistNode,
    Pipeline,
    PipelineNode,
    SaveNode,
)

_CWD_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(_CWD_DIR, "data_outputs", "chinese_fineweb_edu_v2_sft")
DATASET_NAME = "opencsg/Fineweb-Edu-Chinese-V2.3"
DATASET_CONFIG = "messages_no_sys"
DATASET_SPLIT = "train_messages_no_sys"
IGNORE_INDEX = -100

# The generation prompt produced by this template must match the prefix used by
# TokenizeSFTNode. Assistant turns end with eos_token so model.generate can stop.
CHAT_TEMPLATE = """{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% if message['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


class LoadNode(PipelineNode):
    name = "load_sft"

    def __init__(self, take_len: int | None = None, seed: int = 43):
        if take_len is not None and take_len <= 0:
            raise ValueError("take_len 必须大于 0")
        self.take_len = take_len
        self.seed = seed

    def __call__(self, dataset=None) -> IterableDataset:
        dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
            streaming=True,
        )
        if self.take_len is not None:
            dataset = dataset.shuffle(buffer_size=10_000, seed=self.seed)
            dataset = dataset.take(self.take_len)
        return dataset


def _normalize_messages(example: dict[str, Any]) -> list[dict[str, str]] | None:
    raw_messages = example.get("messages_no_sys") or example.get("messages")

    # Also accept the Alpaca subset so changing the source config does not
    # require rewriting the tokenizer stage.
    if raw_messages is None and example.get("instruction") and example.get("output"):
        user_content = str(example["instruction"]).strip()
        extra_input = str(example.get("input") or "").strip()
        if extra_input:
            user_content = f"{user_content}\n\n{extra_input}"
        raw_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": str(example["output"]).strip()},
        ]

    if not isinstance(raw_messages, list):
        return None

    role_aliases = {"human": "user", "gpt": "assistant"}
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            return None
        role = role_aliases.get(message.get("role"), message.get("role"))
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        content = content.strip()
        if content:
            messages.append({"role": role, "content": content})

    if not messages or not any(message["role"] == "assistant" for message in messages):
        return None
    return messages


class TokenizeSFTNode(PipelineNode):
    name = "tokenize_sft"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        context_max_len: int,
    ):
        if context_max_len <= 0:
            raise ValueError("context_max_len 必须大于 0")
        self.tokenizer = tokenizer
        self.context_max_len = context_max_len

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _encode_example(self, example: dict[str, Any]) -> dict[str, list[int]] | None:
        messages = _normalize_messages(example)
        if messages is None:
            return None

        input_ids: list[int] = []
        labels: list[int] = []
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("SFT tokenizer 必须配置 eos_token")

        for message in messages:
            prefix_ids = self._encode(f"<|im_start|>{message['role']}\n")
            content_ids = self._encode(message["content"])
            suffix_ids = self._encode("<|im_end|>\n")

            input_ids.extend(prefix_ids)
            labels.extend([IGNORE_INDEX] * len(prefix_ids))

            input_ids.extend(content_ids)
            if message["role"] == "assistant":
                labels.extend(content_ids)
            else:
                labels.extend([IGNORE_INDEX] * len(content_ids))

            input_ids.extend(suffix_ids)
            if message["role"] == "assistant":
                labels.extend(suffix_ids)
                input_ids.append(eos_id)
                labels.append(eos_id)
            else:
                labels.extend([IGNORE_INDEX] * len(suffix_ids))

        input_ids = input_ids[: self.context_max_len]
        labels = labels[: self.context_max_len]
        if not any(label != IGNORE_INDEX for label in labels):
            return None

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    def __call__(self, dataset: IterableDataset) -> Dataset:
        def _gen() -> Iterator[dict[str, list[int]]]:
            for example in dataset:
                encoded = self._encode_example(example)
                if encoded is not None:
                    yield encoded

        result = Dataset.from_generator(_gen)
        if len(result) == 0:
            raise ValueError("SFT 数据处理后为空，请检查数据字段和 context_max_len")

        supervised_tokens = sum(
            label != IGNORE_INDEX for labels in result["labels"] for label in labels
        )
        total_tokens = sum(len(ids) for ids in result["input_ids"])
        print(
            f"  编码完成: {len(result):,} 条, {total_tokens:,} tokens, "
            f"其中 {supervised_tokens:,} 个 assistant tokens 参与 loss"
        )
        return result


def get_sft_train_dataset(
    tokenizer: PreTrainedTokenizerBase,
    context_max_len: int,
    take_len: int | None = None,
) -> Dataset:
    tokenizer.chat_template = CHAT_TEMPLATE
    sample_tag = str(take_len) if take_len is not None else "all"
    cache_dir = f"{OUTPUT_DIR}_ctx{context_max_len}_n{sample_tag}"
    pipeline = Pipeline(
        [
            CheckExistNode(cache_dir=cache_dir),
            LoadNode(take_len=take_len),
            TokenizeSFTNode(
                tokenizer=tokenizer,
                context_max_len=context_max_len,
            ),
            SaveNode(output_dir=cache_dir),
        ]
    )
    return pipeline.run()


if __name__ == "__main__":
    from learn_llm.model.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    dataset = get_sft_train_dataset(
        tokenizer=tokenizer,
        context_max_len=2048,
        take_len=1000,
    )
    first = dataset[0]
    supervised = [
        token_id
        for token_id, label in zip(first["input_ids"], first["labels"])
        if label != IGNORE_INDEX
    ]
    print(tokenizer.decode(first["input_ids"]))
    print("\n--- supervised assistant text ---")
    print(tokenizer.decode(supervised))
