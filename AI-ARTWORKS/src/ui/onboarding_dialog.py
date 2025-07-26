from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

class OnboardingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to AI-ARTWORK!")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Welcome! Let's get you set up."))
        self.next_btn = QPushButton("Get Started")
        layout.addWidget(self.next_btn)
        self.setLayout(layout)
        # TODO: Add steps for config, model download, UI tour, etc. 