from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QListWidget


class LearningPanel(QWidget):
    """
    Panel for federated learning, feedback submission, and status display.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self.feedback_input = QTextEdit()
        self.submit_feedback = QPushButton("Submit Feedback")
        self.status_list = QListWidget()
        layout.addWidget(QLabel("Submit Feedback:"))
        layout.addWidget(self.feedback_input)
        layout.addWidget(self.submit_feedback)
        layout.addWidget(QLabel("Federated Learning Status:"))
        layout.addWidget(self.status_list)
        # TODO: Connect feedback and status to FederatedLearningManager
