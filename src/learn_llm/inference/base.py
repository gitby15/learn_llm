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
    # model_id = "Qwen/Qwen3-0.6B"
    model_id = "SupraLabs/Supra2-100M"
    # model_id = "Eclipse-Senpai/KeyLM-75M"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, dtype=dtype,
    ).to(device).eval()
    inference = InferenceKernel(model, tokenizer)

    print("交互模式已启动，输入 'exit' 或 'quit' 退出\n")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit"):
            break
        _, output = inference.generate(prompt)
        print(output[len(prompt):])  # 只打印续写部分
        print()

    
    

if __name__ == "__main__":
    test_online()
    # main()
