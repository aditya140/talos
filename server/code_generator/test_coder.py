from coder import CodeGenerator  # type: ignore

code_gen = CodeGenerator()


data = {
    "model": "codegen",
    "prompt": "def hello",
    "suffix": None,
    "max_tokens": 16,
    "temperature": 0.1,
    "top_p": 1.0,
    "n": 1,
    "stream": None,
    "logprobs": None,
    "echo": None,
    "stop": ["\n\n"],
    "presence_penalty": 0,
    "frequency_penalty": 1,
    "best_of": 1,
    "logit_bias": None,
    "user": None,
}

print(code_gen.generate(data))
