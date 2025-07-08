from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QComboBox, QPushButton, QApplication

class AgentExplainDialog(QDialog):
    """
    Dialog for showing agent explanations (XAI).
    Displays agent reasoning, choices, and outputs.
    """
    def __init__(self, agent_registry, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Explanation")
        layout = QVBoxLayout(self)
        self.agent_selector = QComboBox()
        self.agent_selector.addItems(agent_registry.keys())
        self.agent_selector.currentTextChanged.connect(self.update_explanation)
        self.explanation = QTextEdit()
        self.explanation.setReadOnly(True)
        self.copy_btn = QPushButton("Copy Explanation")
        self.copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.explanation.toPlainText()))
        layout.addWidget(QLabel("Select Agent:"))
        layout.addWidget(self.agent_selector)
        layout.addWidget(self.explanation)
        layout.addWidget(self.copy_btn)
        self.setLayout(layout)
        self.agent_registry = agent_registry
        self.update_explanation(self.agent_selector.currentText())

    def update_explanation(self, agent_name):
        agent = self.agent_registry[agent_name]
        doc = agent.__doc__ or "No documentation available."
        self.explanation.setText(doc)
        # TODO: Connect refresh_button to agent.explain() logic 