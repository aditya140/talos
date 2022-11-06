import json
import random
import string
import time

import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore



np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))


class CodeGenerator:
    def __init__(self,verbose:bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        self.model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
        self.MAX_MODEL_LEN = 2048

    class TokensExceedsMaximum(Exception):
        pass


    @staticmethod
    def trim_with_stopwords(output: str, stopwords: list) -> str:
        for w in sorted(stopwords, key=len, reverse=True):
            if output.endswith(w):
                output = output[:-len(w)]
                break
        return output

    @staticmethod
    def to_word_list_format(word_dict, tokenizer):
        flat_ids = []
        offsets = []
        for word_dict_item in word_dict:
            item_flat_ids = []
            item_offsets = []

            for word in word_dict_item:
                ids = tokenizer.encode(word).ids

                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

                # Hack, can we do this better?
                if word == '\n\n':
                    item_flat_ids += [198, 198]
                    item_offsets.append(2)

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))

    def generate(self, data):
        prompt = data['prompt']
        n = data.get('n', 1)
        print(data)
        input_start_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = input_start_ids.shape[1]
        input_len = prompt_len * np.ones([input_start_ids.shape[0], 1]).astype(np.uint32)
        max_tokens = data.get('max_tokens', 16)
        prompt_tokens: int = input_len[0][0]
        requested_tokens = max_tokens + prompt_tokens
        if requested_tokens > self.MAX_MODEL_LEN:
            print(1)
            raise self.TokensExceedsMaximum(
                f"This model's maximum context length is {self.MAX_MODEL_LEN}, however you requested "
                f"{requested_tokens} tokens ({prompt_tokens} in your prompt; {max_tokens} for the completion). "
                f"Please reduce your prompt; or completion length."
            )
        output_len = np.ones_like(input_len).astype(np.uint32) * max_tokens
        num_logprobs = data.get('logprobs', -1)
        if num_logprobs is None:
            num_logprobs = 1
        want_logprobs = num_logprobs > 0

        temperature = data.get('temperature', 0.2)
        if temperature == 0.0:
            temperature = 1.0
            top_k = 1
        else:
            top_k = data.get('top_k', 0)

        top_p = data.get('top_p', 1.0)

        result_ids = self.model.generate(input_start_ids, max_length=128)
        print(result_ids.shape)

        text = self.tokenizer.decode(result_ids[0][prompt_len:], skip_special_tokens=True)

        sequence_lengths = 128
        gen_len = sequence_lengths - input_len.squeeze(1)

        choices = []
        choice = {
            'text': text,
            'index': 1,
            'finish_reason': "stop",
            'logprobs': None,
        }
        choices.append(choice)


        completion = {
            'id': None,  # fill in
            'model': 'codegen',
            'object': 'text_completion',
            'created': int(time.time()),
            'choices': None,  # fill in
            'usage': {
                'completion_tokens': int(gen_len.sum()),
                'prompt_tokens': int(prompt_len),
                'total_tokens': int(gen_len.sum() + prompt_len),
            }
        }
        return completion, choices

    @staticmethod
    def random_completion_id():
        return 'cmpl-' + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(29))

    def streamed_response(self, completion, choices):
        for c in choices:
            completion['id'] = self.random_completion_id()
            completion['choices'] = [c]
            yield f'data: {json.dumps(completion)}\n\n'
        yield 'data: [DONE]\n\n'

    def non_streamed_response(self, completion, choices) -> str:
        completion['id'] = self.random_completion_id()
        completion['choices'] = choices
        return json.dumps(completion)

    def __call__(self, data: dict):
        st = time.time()
        try:
            completion, choices = self.generate(data)
        except:
            completion = {}
            choices = []
        ed = time.time()
        print(f"Returned completion in {(ed - st) * 1000} ms")
        if data.get('stream', False):
            return self.streamed_response(completion, choices)
        else:
            return self.non_streamed_response(completion, choices)