class BaseModel:
    def __init__(self):
        pass

    def user_prompt(self, *prompts, previous_response=None):
        raise Exception('Method not implemented')

    def system_prompt(self, *prompts):
        raise Exception('Method not implemented')