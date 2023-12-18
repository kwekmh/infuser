from backends.engine import BaseBackend


class BasicBackend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.ACTIONS = {
            'wikipedia': self.search_wikipedia,
        }

    def search_wikipedia(self, message, current_try):
        try:
            print(f'Searching Wikipedia for {message}')
            pages = wikipedia.search(message)
            print(f'Pages found: {pages}')
            if current_try < len(pages):
                return f'Observation: {wikipedia.page(sorted(pages)[current_try]).summary}'
        except Exception as e:
            print(e)
            return 'Observation: There was no observation found. Please try again.'

    def query(self, model, message, max_turns=5):
        return model.user_prompt(message)