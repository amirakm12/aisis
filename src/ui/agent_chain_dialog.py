from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QLabel, QHBoxLayout

class AgentChainDialog(QDialog):
    """
    Dialog for building and running custom agent pipelines (chaining).
    Supports both automated and manual chaining.
    """
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