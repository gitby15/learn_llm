import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from typing import TypedDict

class Message(TypedDict):
    role: str
    content: str

class InferenceKernel:
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        print("device", self.model.device)

    def generate(self, messages: list[Message]):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_token = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                # temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            output_str = self.tokenizer.decode(
                output_token[0],
                skip_special_tokens=True,
            )
        return (output_token, output_str)
        


def test():

    # model_id = "BananaMind/BananaMind-2-Mini"
    # model_id = "BananaMind/BananaMind-2.1-Unified"
    # model_id = "Qwen/Qwen3-0.6B"
    model_id = "LiquidAI/LFM2.5-230M"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device).eval()
    inference = InferenceKernel(model, tokenizer)
    prompt = "小孩几岁上学？"
    messages = [
        {"role": "user", "content": prompt}
    ]
    print('=== Prompt: ', prompt)
    _, ourput_str = inference.generate(messages)

    
    print('=== Output: ', ourput_str)

    
    

if __name__ == "__main__":
    test()
