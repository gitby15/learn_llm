import torch
from learn_llm.model.timllm import TimLLM
from learn_llm.model.tokenizer.minimind_tokenizer import MinimindTokenizer


class LLMGenerator:
    def __init__(self, model_path: str = "./trained_model"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.tokenizer = MinimindTokenizer().get_tokenizer()

        self.model = TimLLM.get_exist_model(self.device, model_path)
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

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
