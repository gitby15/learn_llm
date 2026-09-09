import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class InferenceKernel:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()
        print("device", self.model.device)

    def generate(self, prompt: str):
        input_tokens = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output_token = None
        output_str = None
        with torch.no_grad():
            output_token = self.model.generate(
                **input_tokens,
                max_new_tokens=96,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            prompt_length = input_tokens["input_ids"].shape[-1]
            output_str = self.tokenizer.decode(
                output_token[0, prompt_length:],
                skip_special_tokens=True,
            )
        return (output_token, output_str)