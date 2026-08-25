# 使用方法：直接执行当前文件
from transformers import PreTrainedTokenizerFast
from learn_llm._utils_.train_tokenizer import train
from learn_llm.dataset.wikipedia import get_tokenizer_dataset
import os
FOLDER_DIR = os.path.dirname(os.path.abspath(__file__))

def get_tokenizer():
    if not os.path.exists(os.path.join(FOLDER_DIR, "tokenizer.json")):
        raise ValueError("tokenizer未训练，请先训练分词器")
    return PreTrainedTokenizerFast.from_pretrained(FOLDER_DIR)
    

if __name__ == "__main__":
    dataset = get_tokenizer_dataset()
    train(FOLDER_DIR, dataset)

    