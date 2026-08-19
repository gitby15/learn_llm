import os

from transformers import PreTrainedModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ModelPath:
    BASE = os.path.join(_PROJECT_ROOT, "model_outputs")

    PRETRAIN_CHECKPOINT = os.path.join(BASE, "pretrain", "checkpoints")
    PRETRAIN_SAVE = os.path.join(BASE, "pretrain", "save")

    SFT_CHECKPOINT = os.path.join(BASE, "sft", "checkpoints")
    SFT_SAVE = os.path.join(BASE, "sft", "save")

    @staticmethod
    def get_exist_model(model_class: PreTrainedModel, model_dir: str):
        try:
            print(f"尝试加载模型，类型： {model_class.__name__} | 路径： {model_dir}")
            return model_class.from_pretrained(model_dir)
        except:
            print(f"模型 {model_dir} 不存在, 返回空")
            return None
