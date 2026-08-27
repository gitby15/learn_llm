import os

from transformers import PreTrainedModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ModelPath:
    BASE = os.path.join(_PROJECT_ROOT, "model_outputs")

    PRETRAIN_CHECKPOINT = os.path.join(BASE, "pretrain", "checkpoints")
    PRETRAIN_SAVE = os.path.join(BASE, "pretrain", "save")

    SFT_CHECKPOINT = os.path.join(BASE, "sft", "checkpoints")
    SFT_SAVE = os.path.join(BASE, "sft", "save")

    TOKENIZER = os.path.join(BASE, "model", "tokenizer")

    @staticmethod
    def get_exist_model(model_class: type[PreTrainedModel], model_dir: str) -> PreTrainedModel | None:
        try:
            print(f"尝试加载模型，类型： {model_class.__name__} | 路径： {model_dir}")
            return model_class.from_pretrained(model_dir)
        except:
            print(f"模型 {model_dir} 不存在, 返回空")
            return None

    @staticmethod
    def get_latest_checkpoint_model(model_class: type[PreTrainedModel], checkpoint_dir: str) -> PreTrainedModel | None:
        """从 checkpoint 目录中自动找到最新的 checkpoint 并加载模型"""
        if not os.path.isdir(checkpoint_dir):
            return None
        checkpoints = sorted(
            [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        if not checkpoints:
            return None
        latest = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"加载最新 checkpoint: {latest}")
        return model_class.from_pretrained(latest)