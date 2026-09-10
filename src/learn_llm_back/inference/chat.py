import argparse
import os
from typing import Literal, TypedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from learn_llm._utils_.model_path import ModelPath
from learn_llm.dataset.chinese_fineweb.sft import CHAT_TEMPLATE


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatSession:
    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizerBase,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        system_prompt: str | None = None,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.context_length = model.config.max_position_embeddings

        if max_new_tokens <= 0 or max_new_tokens >= self.context_length:
            raise ValueError(
                f"max_new_tokens 必须在 1 到 {self.context_length - 1} 之间"
            )
        if temperature < 0:
            raise ValueError("temperature 不能小于 0")
        if not 0 < top_p <= 1:
            raise ValueError("top_p 必须在 (0, 1] 范围内")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty 必须大于 0")

        if not self.tokenizer.chat_template:
            self.tokenizer.chat_template = CHAT_TEMPLATE

        self._system_prompt = system_prompt.strip() if system_prompt else None
        self.messages: list[Message] = []
        self.clear()

    def clear(self) -> None:
        self.messages = []
        if self._system_prompt:
            self.messages.append(
                {"role": "system", "content": self._system_prompt}
            )

    def _tokenize_messages(self, messages: list[Message]):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def _fit_context(self, messages: list[Message]):
        input_limit = self.context_length - self.max_new_tokens
        fitted = list(messages)
        model_inputs = self._tokenize_messages(fitted)

        # Keep the optional system message and current user message. Remove
        # complete old user/assistant turns until the prompt fits.
        while model_inputs["input_ids"].shape[-1] > input_limit:
            first_history_index = (
                1 if fitted and fitted[0]["role"] == "system" else 0
            )
            if len(fitted) - first_history_index <= 1:
                break

            remove_count = 1
            if (
                first_history_index + 1 < len(fitted) - 1
                and fitted[first_history_index + 1]["role"] == "assistant"
            ):
                remove_count = 2
            del fitted[first_history_index : first_history_index + remove_count]
            model_inputs = self._tokenize_messages(fitted)

        if model_inputs["input_ids"].shape[-1] > input_limit:
            # A single user message can itself exceed the context window.
            # Left truncation preserves its ending and the assistant prefix.
            for key in ("input_ids", "attention_mask"):
                if key in model_inputs:
                    model_inputs[key] = model_inputs[key][:, -input_limit:]

        return fitted, model_inputs

    def ask(self, content: str) -> str:
        content = content.strip()
        if not content:
            raise ValueError("消息不能为空")

        pending_messages = [
            *self.messages,
            {"role": "user", "content": content},
        ]
        fitted_messages, model_inputs = self._fit_context(pending_messages)
        model_inputs = model_inputs.to(self.model.device)
        prompt_length = model_inputs["input_ids"].shape[-1]

        eos_token_ids = [self.tokenizer.eos_token_id]
        im_end_id = (
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            if "<|im_end|>" in self.tokenizer.get_vocab()
            else None
        )
        if (
            isinstance(im_end_id, int)
            and im_end_id >= 0
            and im_end_id not in eos_token_ids
        ):
            eos_token_ids.append(im_end_id)
        eos_token_ids = [token_id for token_id in eos_token_ids if token_id is not None]
        if not eos_token_ids:
            raise ValueError("tokenizer 必须配置 eos_token 或 <|im_end|>")

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_ids[0]

        generation_args = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_ids,
            "use_cache": True,
        }
        if self.temperature > 0:
            generation_args["temperature"] = self.temperature
            generation_args["top_p"] = self.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**model_inputs, **generation_args)

        response = self.tokenizer.decode(
            output_ids[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()
        self.messages = [
            *fitted_messages,
            {"role": "assistant", "content": response},
        ]
        return response


def _latest_sft_model_path() -> str:
    if os.path.isfile(os.path.join(ModelPath.SFT_SAVE, "config.json")):
        return ModelPath.SFT_SAVE

    checkpoint_dir = ModelPath.SFT_CHECKPOINT
    if os.path.isdir(checkpoint_dir):
        checkpoints = [
            name
            for name in os.listdir(checkpoint_dir)
            if name.startswith("checkpoint-") and name.split("-")[-1].isdigit()
        ]
        if checkpoints:
            latest = max(checkpoints, key=lambda name: int(name.split("-")[-1]))
            return os.path.join(checkpoint_dir, latest)

    raise FileNotFoundError(
        "没有找到 SFT 模型，请先运行 `uv run sft-t` 完成训练"
    )


def load_chat_session(
    model_path: str | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    system_prompt: str | None = None,
) -> ChatSession:
    model_path = model_path or _latest_sft_model_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    print(f"模型: {model_path}")
    print(f"设备: {device}, dtype: {dtype}")

    return ChatSession(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        system_prompt=system_prompt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TimLLM 多轮对话")
    parser.add_argument("--model", help="本地模型目录，默认使用 SFT 模型")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--system", help="可选的 system prompt")
    args = parser.parse_args()

    chat = load_chat_session(
        model_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        system_prompt=args.system,
    )

    print("输入 /clear 清空上下文，/history 查看上下文，exit 或 quit 退出")
    while True:
        try:
            content = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if not content:
            continue
        if content.lower() in {"exit", "quit"}:
            break
        if content == "/clear":
            chat.clear()
            print("上下文已清空")
            continue
        if content == "/history":
            for message in chat.messages:
                print(f"{message['role']}: {message['content']}")
            continue

        try:
            print(f"助手: {chat.ask(content)}")
        except RuntimeError as error:
            if "out of memory" in str(error).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"生成失败: {error}")


if __name__ == "__main__":
    main()
