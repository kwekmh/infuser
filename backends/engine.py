from models.model import BaseModel


class BaseBackend:
    def __init__(self):
        pass

    def query(self, model, message):
        raise Exception('Method not implemented')

class Engine:
    def __init__(self):
        self._model = None
        self._backend = None

    def load_model(self, model: BaseModel):
        self._model = model

    def load_backend(self, backend: BaseBackend):
        self._backend = backend

    def query(self, prompt: str):
        if self._backend:
            return self._backend.query(self._model, prompt)
        else:
            raise Exception("Backend not loaded")