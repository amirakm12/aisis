from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton, QTextEdit

class LearningPanel(QWidget):
    """
    Panel for federated learning, feedback submission, and status display.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.status_label = QLabel("Learning Status: Idle")
        self.progress = QProgressBar()
        self.metrics = QTextEdit()
        self.metrics.setReadOnly(True)
        self.feedback = QTextEdit()
        self.feedback.setPlaceholderText("Enter feedback for the agent/model...")
        self.retrain_btn = QPushButton("Retrain")
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(QLabel("Recent Metrics:"))
        layout.addWidget(self.metrics)
        layout.addWidget(QLabel("User Feedback:"))
        layout.addWidget(self.feedback)
        layout.addWidget(self.retrain_btn)
        self.setLayout(layout)
        # TODO: Connect retrain_btn to retraining logic, update progress/metrics from backend 