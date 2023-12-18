from backends.engine import BaseBackend


class BasicBackend(BaseBackend):
    def __init__(self):
        super().__init__()

    def query(self, model, message, max_turns=5):
        return model.user_prompt(message)