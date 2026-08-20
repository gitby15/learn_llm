import argparse
from learn_llm.inference import LLMGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description="learn-llm 推理")
    parser.add_argument("--model", help="模型路径")
    parser.add_argument("--prompt", default=None, help="单次推理的 prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--chat", action="store_true", help="使用 chat_template 包装输入（SFT 模型推荐开启）")
    args = parser.parse_args()

    generator = LLMGenerator(model_path=args.model, use_chat_template=args.chat)

    if args.prompt:
        output = generator.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(output)
    else:
        print("进入交互模式，输入 'exit' 退出")
        while True:
            try:
                prompt = input("\n>>> ")
            except (EOFError, KeyboardInterrupt):
                print("手动退出")
                break
            if prompt.strip().lower() == "exit":
                break
            if not prompt.strip():
                continue
            output = generator.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            print(output)
