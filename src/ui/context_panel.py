from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QTextEdit
from PyQt6.QtCore import Qt
from .context_manager import ContextManager

class ContextPanel(QWidget):
    """
    Context Panel for workspace context management, progress, and state display.
    Integrates with ContextManager for adaptive UI/UX.
    """
    def __init__(self, context_manager: ContextManager, parent=None):
        super().__init__(parent)
        self.context_manager = context_manager
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title_label = QLabel("Workspace Context")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)

        self.context_display = QTextEdit()
        self.context_display.setReadOnly(True)
        layout.addWidget(self.context_display)

        self.progress_label = QLabel("Progress:")
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.state_label = QLabel("State: Idle")
        layout.addWidget(self.state_label)

        self.update_context_display()

    def update_context_display(self):
        context = self.context_manager.session_context
        text = "\n".join(f"{k}: {v}" for k, v in context.items())
        self.context_display.setText(text)

    def set_progress(self, value: int, text: str = None):
        self.progress_bar.setValue(value)
        if text:
            self.progress_label.setText(f"Progress: {text}")

    def set_state(self, state: str):
        self.state_label.setText(f"State: {state}")

    def update_context(self, event):
        self.context_manager.update_context(event)
        self.update_context_display() 