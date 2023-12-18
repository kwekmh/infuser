import argparse
from enum import Enum

from backends.basic.core import BasicBackend
from backends.engine import Engine
from backends.react.core import ReactBackend
from models.llama.huggingface import LlamaHuggingFace
from models.openai.chatgpt import ChatGPTModel
from dotenv import dotenv_values


class BackendType(Enum):
    BASIC = 1
    REACT = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_string(s):
        try:
            return BackendType[s]
        except KeyError:
            return s


class ModelType(Enum):
    CHATGPT = 1
    LLAMA = 2

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    @staticmethod
    def from_string(s):
        try:
            return ModelType[s]
        except KeyError:
            return s


def main():
    env_config = dotenv_values('.env')

    parser = argparse.ArgumentParser(
        prog='Infuser',
        description='LLM agent'
    )

    parser.add_argument('-b', '--backend', type=BackendType.from_string, choices=list(BackendType), required=True)
    parser.add_argument('-p', '--prompt', type=str, required=True)
    parser.add_argument('-m', '--model', type=ModelType.from_string, choices=list(ModelType), required=True)

    args = parser.parse_args()

    model = None
    backend = None

    if args.model == ModelType.CHATGPT:
        model = ChatGPTModel(env_config['OPENAI_API_KEY'])
    elif args.model == ModelType.LLAMA:
        model = LlamaHuggingFace(env_config['LLAMA_MODEL_PATH'])

    if args.backend == BackendType.BASIC:
        backend = BasicBackend()
    elif args.backend == BackendType.REACT:
        backend = ReactBackend()
    engine = Engine()

    engine.load_model(model)
    engine.load_backend(backend)

    prompt = args.prompt

    print('Prompt: {}'.format(prompt))
    print('Response: {}'.format(engine.query(f'Question: {prompt}')))


if __name__ == '__main__':
    main()
