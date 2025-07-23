from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QLabel, QHBoxLayout
from PySide6.QtStateMachine import QStateMachine, QState
from PySide6.QtCore import Signal

class AgentChainDialog(QDialog):
    """
    Dialog for building and running custom agent pipelines (chaining).
    Supports both automated and manual chaining.
    """
    agent_finished = Signal()

    def __init__(self, agent_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Pipeline Builder")
        layout = QVBoxLayout(self)
        self.available_agents = QListWidget()
        self.available_agents.addItems(agent_names)
        self.pipeline = QListWidget()
        self.add_button = QPushButton("Add to Pipeline")
        self.remove_button = QPushButton("Remove from Pipeline")
        self.run_button = QPushButton("Run Pipeline")
        layout.addWidget(QLabel("Available Agents:"))
        layout.addWidget(self.available_agents)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.add_button)
        btn_layout.addWidget(self.remove_button)
        layout.addLayout(btn_layout)
        layout.addWidget(QLabel("Pipeline Order:"))
        layout.addWidget(self.pipeline)
        layout.addWidget(self.run_button)
        # TODO: Connect buttons to pipeline logic
        self.run_button.clicked.connect(self.run_pipeline)

    def run_pipeline(self):
        pipeline = [self.pipeline.item(i).text() for i in range(self.pipeline.count())]
        if not pipeline:
            return
        self.machine = QStateMachine(self)
        previous = None
        for agent in pipeline:
            state = QState()
            state.entered.connect(lambda a=agent: self._execute_agent(a))
            if previous:
                previous.addTransition(self.agent_finished, state)
            else:
                self.machine.setInitialState(state)
            previous = state
        self.machine.start()

    def _execute_agent(self, agent_name):
        # TODO: Execute the agent asynchronously and emit agent_finished when done
        pass 