from openai import OpenAI

from models.model import BaseModel


class ChatGPTModel(BaseModel):
    def __init__(self, api_key: str):
        super().__init__()
        self._api_key = api_key
        self._client = OpenAI(api_key=self._api_key)
        self._model = 'gpt-4'
        self._system_prompts = []
        self._user_prompts = []

    def system_prompt(self, *prompts):
        messages = []
        for prompt in prompts:
            self._system_prompts.append({'role': 'system', 'content': prompt})
        return self._system_prompts

    def user_prompt(self, *prompts, previous_response=None):
        print('in user prompt with {}'.format(prompts))
        for prompt in prompts:
            self._user_prompts.append({'role': 'user', 'content': prompt})
        messages = self._system_prompts + self._user_prompts
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages
        )
        print(response)
        print(response.usage)
        return response.choices[0].message.content
