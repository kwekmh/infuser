from models.model import BaseModel
import torch
import transformers

from transformers import AutoModelForCausalLM, LlamaTokenizer


class LlamaHuggingFace(BaseModel):
    def __init__(self, model_dir):
        super().__init__()
        self._model_dir = model_dir
        self._model = AutoModelForCausalLM.from_pretrained(self._model_dir)
        self._tokenizer = LlamaTokenizer.from_pretrained(self._model_dir)
        self._pipeline = transformers.pipeline(
            'text-generation',
            model=self._model,
            tokenizer=self._tokenizer,
            torch_dtype=torch.float16,
            device=0,
        )
        self._system_prompts = []
        self._user_prompts = []
        self._final_prompts = [f'{self._tokenizer.bos_token}[INST]']

    def system_prompt(self, *prompts):
        for prompt in prompts:
            self._system_prompts.append({'role': 'system', 'content': prompt})
        return self._system_prompts

    def user_prompt(self, *prompts, previous_response=None):
        if previous_response:
            self._user_prompts.append({'role': 'assistant', 'content': previous_response})
        for prompt in prompts:
            self._user_prompts.append({'role': 'user', 'content': prompt})
        final_prompt = self._tokenizer.apply_chat_template(self._system_prompts + self._user_prompts, tokenize=False, add_generation_prompt=True)
        print(f'Final prompt: {final_prompt}')
        print('Staring pipeline')
        sequences = self._pipeline(
            final_prompt,
            do_sample=True,
            top_k=10,
            eos_token_id=self._tokenizer.eos_token_id,
            return_full_text=False,
            max_length=4096,)
        print('Pipeline complete')
        print(f'Sequences: {sequences}')
        for seq in sequences:
            print(f'Sequence:\n{seq}')
            print(f'Generated text:\n{seq["generated_text"]}')
        return '\n'.join([s['generated_text'] for s in sequences])