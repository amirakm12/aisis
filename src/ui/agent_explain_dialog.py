from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit, QPushButton


class AgentExplainDialog(QDialog):
    """
    Dialog for showing agent explanations (XAI).
    Displays agent reasoning, choices, and outputs.
    """

    def __init__(self, agent_name, explanation, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Explain Agent: {agent_name}")
        layout = QVBoxLayout(self)
        self.explanation_box = QTextEdit()
        self.explanation_box.setReadOnly(True)
        self.explanation_box.setText(explanation)
        self.refresh_button = QPushButton("Refresh Explanation")
        layout.addWidget(QLabel("Explanation:"))
        layout.addWidget(self.explanation_box)
        layout.addWidget(self.refresh_button)
        # TODO: Connect refresh_button to agent.explain() logic
