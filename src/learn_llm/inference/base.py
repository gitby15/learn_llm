import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from learn_llm._utils_.model_path import ModelPath
from learn_llm.inference.kernel import InferenceKernel

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        ModelPath.PRETRAIN_SAVE,
        trust_remote_code=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        ModelPath.PRETRAIN_SAVE,
        trust_remote_code=True,
    )
    
    inference = InferenceKernel(model, tokenizer)
    prompt = "番茄炒蛋"
    print('=== Prompt: ', prompt)
    _, output_str = inference.generate(prompt)
    
        
    print('=== Output: ', output_str)




def test_online():

    # model_id = "BananaMind/BananaMind-2-Mini"
    # model_id = "BananaMind/BananaMind-2.1-Unified"
    model_id = "Qwen/Qwen3-0.6B"

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
    prompt = "今天的天气"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print('=== Prompt: ', prompt)
    _, ourput_str = inference.generate(prompt)

    
    print('=== Output: ', ourput_str)

    
    

if __name__ == "__main__":
    # test()
    main()
