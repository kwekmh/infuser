from backends.engine import BaseBackend
import wikipedia
class ReactBackend(BaseBackend):
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
        react_prompt = '''
You run in a loop of Thought, Action, PAUSE and Observation.

At the end of every loop, you output an Answer.

Use Thought to describe your thoughts about the Question you have been asked.

Use Action to run one of the actions available to you. Return PAUSE after each Action.

When you PAUSE, you do not Answer, and you wait for an Observation.

Observation is the output from each Action that will be given to you.

You combine Observation with your own understanding to attempt to answer the question.

The actions available to you are:

wikipedia:
e.g. wikipedia: python
Returns a summary from looking up Wikipedia.

Always look things up on Wikipedia if you can.

Example session:

Question: What is the capital city of Indonesia?
Thought: I should look up Indonesia on Wikipedia.
Action: wikipedia: Indonesia
PAUSE

You will be called again with:

Observation: Indonesia,[a] officially the Republic of Indonesia,[b] is a country in Southeast Asia and Oceania between the Indian and Pacific oceans. The country's capital, Jakarta, is the world's second-most populous urban area.

You will then answer, if you have the answer:

Answer: The capital city of Indonesia is Jakarta.

If you do not have the answer, you can return an Action and return PAUSE after.
        '''.strip()

        i = 0
        model.system_prompt(react_prompt)
        response = model.user_prompt(message)
        while i < max_turns:
            i += 1
            lines = response.splitlines()
            print(f'Lines: {lines}')
            answers = [' '.join(v.split()[1:]) for v in lines if v.startswith('Answer:')
                       for v_split in [v.split()] if len(v_split) > 1]
            if len(answers) > 0:
                return answers
            actions = [[v.split()[1], ' '.join(v.split()[2:])] for v in lines if v.startswith('Action:')
                       for v_split in [v.split()] if len(v_split) > 2]
            if len(actions) > 0:
                print(f'Actions: {actions}')
                messages = [self.ACTIONS[action[:-1]](value, i) for action, value in actions]
                response = model.user_prompt(*messages, previous_response=response)
            else:
                return response
        return response
