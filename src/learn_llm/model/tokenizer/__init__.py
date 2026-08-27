import os
from transformers import PreTrainedTokenizerFast

FOLDER_DIR = os.path.dirname(os.path.abspath(__file__))

def _check_exist_tokenizer() -> bool:
    result = True
    if not os.path.exists(os.path.join(FOLDER_DIR, "tokenizer.json")):
        result = False
    if not os.path.exists(os.path.join(FOLDER_DIR, "tokenizer_config.json")):
        result = False

    return result


def get_tokenizer():
    if _check_exist_tokenizer() is False:
        raise ValueError("tokenizer未训练，请先训练分词器")
    return PreTrainedTokenizerFast.from_pretrained(FOLDER_DIR)
