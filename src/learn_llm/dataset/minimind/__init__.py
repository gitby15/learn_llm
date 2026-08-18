import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from modelscope.hub.file_download import dataset_file_download

def _download_data():
    file_path = dataset_file_download(
        dataset_id='gongjy/minimind_dataset',
        file_path='pretrain_t2t_mini.jsonl'
    )
    print(f"Minimind Dataset 文件保存路径: {file_path}")
    return file_path

# 是一个很大的jsonl
def get_train_dataset():
    data_files = _download_data()
    return load_dataset('json', data_files=data_files, split='train', streaming=True)
