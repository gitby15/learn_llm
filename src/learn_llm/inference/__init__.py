import torch
from learn_llm._utils_.model_path import ModelPath
from learn_llm.model.timllm import TimLLM
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer


class LLMGenerator:
    def __init__(self, model_path: str, use_chat_template: bool = False):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.tokenizer = MinimindTokenizer.get_tokenizer()
        self.use_chat_template = use_chat_template

        self.model = ModelPath.get_exist_model(TimLLM, model_path)
        if self.model is None:
            raise FileNotFoundError(f"模型 {model_path} 不存在")
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        # 这两个参数是用于重复惩罚的，Todo：弄清楚原理
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        if self.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt

        max_input_len = self.model.config.max_position_embeddings - max_new_tokens
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated

    def generate_chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 1.3,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        max_input_len = self.model.config.max_position_embeddings - max_new_tokens
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated
