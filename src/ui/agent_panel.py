from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QListWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QHBoxLayout,
    QCheckBox,
)


class AgentPanel(QWidget):
    """
    Panel for listing, selecting, invoking, chaining, and explaining agents.
    Default: automated agent selection/chaining, but user can override.
    """

    def __init__(self, orchestrator, parent=None):
        super().__init__(parent)
        self.orchestrator = orchestrator
        layout = QVBoxLayout(self)
        self.automation_checkbox = QCheckBox("Automated (recommended)")
        self.automation_checkbox.setChecked(True)
        self.agent_list = QListWidget()
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.invoke_button = QPushButton("Invoke Agent")
        self.chain_button = QPushButton("Chain Agents")
        self.explain_button = QPushButton("Explain Agent")
        layout.addWidget(self.automation_checkbox)
        layout.addWidget(QLabel("Available Agents:"))
        layout.addWidget(self.agent_list)
        layout.addWidget(QLabel("Agent Info:"))
        layout.addWidget(self.info_box)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.invoke_button)
        btn_layout.addWidget(self.chain_button)
        btn_layout.addWidget(self.explain_button)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.populate_agents()
        self.agent_list.currentItemChanged.connect(self.show_agent_info)
        # TODO: Connect buttons to agent invocation, chaining, and explainability logic
        # TODO: If automation_checkbox is checked, use orchestrator's default pipeline

    def populate_agents(self):
        self.agent_list.clear()
        for name in self.orchestrator.agents.keys():
            self.agent_list.addItem(name)

    def show_agent_info(self, current, previous):
        if current:
            agent = self.orchestrator.agents[current.text()]
            self.info_box.setText(str(agent.__doc__ or agent.__class__.__name__))
