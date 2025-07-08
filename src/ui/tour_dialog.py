from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton

class TourDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome Tour")
        layout = QVBoxLayout(self)
        self.steps = [
            "Welcome to AISIS!",
            "This is the main window where you can load and edit images.",
            "Use the left sidebar to access agents, plugins, and settings.",
            "The chat panel lets you interact with the AI and refine workflows.",
            "Access the model zoo to download and switch AI models.",
            "Visit settings to customize your experience.",
            "You're ready to get started!"
        ]
        self.current_step = 0
        self.label = QLabel(self.steps[self.current_step])
        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_step)
        layout.addWidget(self.label)
        layout.addWidget(self.next_btn)
        self.setLayout(layout)

    def next_step(self):
        self.current_step += 1
        if self.current_step < len(self.steps):
            self.label.setText(self.steps[self.current_step])
        else:
            self.accept() 