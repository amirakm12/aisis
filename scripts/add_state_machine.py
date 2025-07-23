import argparse
import os

TEMPLATE = '''
from .base_agent import BaseAgent
from typing import Dict

class {class_name}(BaseAgent):
    def __init__(self):
        super().__init__()
        self.current_state = 'initial'
        self.transitions: Dict[str, Dict[str, str]] = {{
            'initial': {{'start': 'processing'}},
            'processing': {{'complete': 'done', 'error': 'error'}},
            'done': {{}},
            'error': {{'retry': 'processing'}},
        }}

    def transition(self, action: str) -> bool:
        if action in self.transitions.get(self.current_state, {{}}):
            self.current_state = self.transitions[self.current_state][action]
            return True
        return False

    def execute(self, input_data):
        # TODO: Implement agent logic using state machine
        if self.transition('start'):
            # Processing logic
            try:
                result = self.process(input_data)
                self.transition('complete')
            except Exception:
                self.transition('error')
            return result
        return None

    def process(self, input_data):
        # Placeholder for actual processing
        return input_data

# Register the agent
register_agent('{agent_name}', {class_name}())
'''


def main():
    parser = argparse.ArgumentParser(
        description='Add a new state machine-based agent to the project.'
    )
    parser.add_argument('name', help='Name of the new agent (e.g., NewAgent)')
    args = parser.parse_args()

    class_name = args.name.capitalize() + 'Agent'
    agent_name = args.name.lower()
    file_name = f'{agent_name}_agent.py'
    file_path = os.path.join('src', 'agents', file_name)

    if os.path.exists(file_path):
        print(f'Error: File {file_path} already exists.')
        return

    content = TEMPLATE.format(class_name=class_name, agent_name=agent_name)

    with open(file_path, 'w') as f:
        f.write(content)

    print(f'Created new agent file: {file_path}')
    print('Remember to import it in your main code if needed.')


if __name__ == '__main__':
    main()
