from PyQt6.QtCore import QStateMachine, QState, QFinalState
from PyQt6.QtWidgets import QMessageBox

class WorkflowStateMachine(QStateMachine):
    def __init__(self, workflow: list[str], orchestrator, input_data, parent=None):
        super().__init__(parent)
        self.workflow = workflow
        self.orchestrator = orchestrator
        self.input_data = input_data
        self.current_result = input_data
        self.states = []
        self.setup_states()

    def setup_states(self):
        initial_state = QState()
        initial_state.entered.connect(lambda: self.run_agent(0))
        self.addState(initial_state)
        self.setInitialState(initial_state)
        self.states.append(initial_state)

        for i in range(1, len(self.workflow)):
            state = QState()
            state.entered.connect(lambda idx=i: self.run_agent(idx))
            self.addState(state)
            self.states.append(state)

        final_state = QFinalState()
        self.addState(final_state)
        self.states[-1].addTransition(self.finished, final_state)

        for i in range(len(self.states) - 1):
            self.states[i].addTransition(self.finished, self.states[i + 1])

        self.finished.connect(self.on_workflow_complete)

    def run_agent(self, index: int):
        agent_name = self.workflow[index]
        try:
            self.current_result = self.orchestrator.execute_single_agent(agent_name, self.current_result)
        except Exception as e:
            QMessageBox.warning(None, "Error", f"Error in agent {agent_name}: {str(e)}")
            self.stop()

    def on_workflow_complete(self):
        QMessageBox.information(None, "Workflow Complete", "The workflow has completed successfully.")
        # Handle final result
        print(self.current_result)