import os
import json
from bottle import Bottle, request, static_file, response

from learn_llm._utils_.model_path import ModelPath
from learn_llm.inference import LLMGenerator

app = Bottle()
_generator: LLMGenerator | None = None

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def get_generator() -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator(model_path=ModelPath.SFT_SAVE, use_chat_template=True)
    return _generator


@app.get("/")
def index():
    return static_file("index.html", root=STATIC_DIR)


@app.get("/static/<filename:path>")
def serve_static(filename: str):
    return static_file(filename, root=STATIC_DIR)


@app.post("/generate")
def generate():
    data = request.json
    messages = data.get("messages", [])
    if not messages:
        response.status = 400
        return {"error": "messages 不能为空"}

    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 0.7)

    generator = get_generator()
    output = generator.generate_chat(
        messages,
        max_new_tokens=max_tokens,
        temperature=temperature,
    )
    return {"generated": output}


def main():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    main()