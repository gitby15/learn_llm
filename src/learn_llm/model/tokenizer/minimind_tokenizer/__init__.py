# 暂时先直接用minimnd的Tokenizer，先跑通流程，再回来啃这一块

import os
from transformers import AutoTokenizer

folder_path = os.path.dirname(__file__)


class MinimindTokenizer:
    _tokenizer = None
    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            cls._tokenizer = AutoTokenizer.from_pretrained(folder_path)
        return cls._tokenizer


if __name__ == "__main__":
    tokenizer_1 = MinimindTokenizer.get_tokenizer()
    tokenizer_2 = MinimindTokenizer.get_tokenizer()
    print(tokenizer_1)
    print(tokenizer_2)