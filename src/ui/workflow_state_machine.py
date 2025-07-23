"""
from PySide6.QtCore import QStateMachine, QState, QFinalState, Signal
from PySide6.QtWidgets import QApplication

from ..agents.multi_agent_orchestrator import MultiAgentOrchestrator

class WorkflowStateMachine(QStateMachine):
    \"\"\"State machine for mapping agent workflows with seamless transitions.\"\"\"
    taskCompleted = Signal(dict)
    
    def __init__(self, orchestrator: MultiAgentOrchestrator, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        
    def setup_workflow(self, workflow: list[str]):
        \"\"\"Setup states for a given workflow (list of agent names).\"\"\"
        previous_state = None
        for agent_name in workflow:
            state = QState(self)
            state.entered.connect(lambda a=agent_name: self.run_agent(a))
            if previous_state is None:
                self.setInitialState(state)
            else:
                previous_state.addTransition(state)
            previous_state = state
        final = QFinalState(self)
        previous_state.addTransition(final)
        final.entered.connect(self.on_finished)
    
    def run_agent(self, agent_name: str):
        \"\"\"Run the agent asynchronously.\"\"\"
        # In practice, use QThread or async to run
        result = self.orchestrator.delegate_task({}, [agent_name])
        self.taskCompleted.emit(result)
    
    def on_finished(self):
        print(\"Workflow completed\")
"""