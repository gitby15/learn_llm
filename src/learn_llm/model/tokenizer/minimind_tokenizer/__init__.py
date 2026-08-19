# 暂时先直接用minimnd的Tokenizer，先跑通流程，再回来啃这一块

import os
from transformers import AutoTokenizer

folder_path = os.path.dirname(__file__)

class MinimindTokenizer(AutoTokenizer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(folder_path)
    def get_tokenizer(self):
        return self.tokenizer


if __name__ == "__main__":
    tokenizer = MinimindTokenizer()
    print(tokenizer.get_pad_id())
    print(tokenizer.get_vocab_size())