import re
from typing import List

# Read the file
with open('src/ui/main_window.py', 'r') as f:
    lines = f.readlines()

# Find the last QtGui import
last_gui_index = -1
for i, line in enumerate(lines):
    if 'from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent' in line:
        last_gui_index = i
        break

if last_gui_index != -1:
    new_import = "from PySide6.QtStateMachine import QStateMachine, QState" + "
"
    lines = lines[:last_gui_index+1] + [new_import] + lines[last_gui_index+1:]

# Append the new methods
new_methods = '''
    def start_workflow(self, workflow: List[str]) -> None:
        \"\"\"Start a chained workflow using state machine for seamless transitions.\"\"\"
        self.state_machine = QStateMachine(self)
        previous = None
        states = []
        for agent_name in workflow:
            state = QState()
            state.entered.connect(lambda a=agent_name: self.run_workflow_step(a))
            if previous:
                previous.addTransition(self.operation_complete, state)
            states.append(state)
            self.state_machine.addState(state)
            previous = state
        final_state = QState()
        previous.addTransition(self.operation_complete, final_state)
        final_state.entered.connect(self.workflow_complete)
        self.state_machine.addState(final_state)
        self.state_machine.setInitialState(states[0])
        self.state_machine.start()

    def run_workflow_step(self, agent_name: str) -> None:
        task = {'type': agent_name, 'image': self.current_image}
        worker = AsyncWorker(self.orchestrator.delegate_task, task, [agent_name])
        worker.finished.connect(lambda result: self.operation_complete.emit())
        worker.start()

    def workflow_complete(self):
        QMessageBox.information(self, \"Workflow Complete\", \"The workflow has completed successfully.\")
'''
lines.append(new_methods)

# Write back
with open('src/ui/main_window.py', 'w') as f:
    f.writelines(lines)
print('Added state machine workflow to main_window.py')

