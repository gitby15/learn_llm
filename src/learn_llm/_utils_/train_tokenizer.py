from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast
from datasets import Dataset


SPECIAL_TOKEN = [
    "<|endoftext|>",   # EOS / 文档结束
    "<|pad|>",          # 填充
    "<|im_start|>",     # 对话轮次开始
    "<|im_end|>",       # 对话轮次结束
]

def train(
        folder_dir: str,
        dataset: Dataset,
        vocab_size: int = 6480,
        min_frequency:int = 10,
):
    tokenizer = Tokenizer(models.BPE())
    # 关键：ByteLevel pre-tokenizer 会把所有文本映射到字节层面
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    # ByteLevel 解码器，和 pre_tokenizer 配套使用
    tokenizer.decoder = decoders.ByteLevel()
    print(f"开始训练，训练参数：vocab_size={vocab_size}, min_frequency={min_frequency}")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKEN,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def _iter():
        for item in dataset:
            text = item['text']
            yield text

    gen = _iter()
    try:
        tokenizer.train_from_iterator(gen, trainer)
    finally:
        gen.close()

    # 包装为 PreTrainedTokenizerFast，这样保存后会生成：
    #   - tokenizer.json          (核心词表/merges)
    #   - tokenizer_config.json   (配置元信息)
    #   - special_tokens_map.json (特殊 token 映射)
    #   可以用 AutoTokenizer.from_pretrained() 直接加载
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    wrapped.save_pretrained(folder_dir)
    # 打印词表长度、训练时间、训练数据量
    print(f"词表长度: {len(wrapped.get_vocab())}")
    try:
        print(f"训练数据量: {len(dataset)}")
    except TypeError:
        pass
    
