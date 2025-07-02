from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QTextEdit, QPushButton, QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from .context_manager import ContextManager

class ContextPanel(QWidget):
    """
    Enhanced Context Panel for workspace context, progress, state, and logs.
    Integrates with ContextManager for adaptive UI/UX.
    """
    context_updated = pyqtSignal(dict)
    progress_updated = pyqtSignal(int, str)
    state_changed = pyqtSignal(str)

    def __init__(self, context_manager: ContextManager, parent=None):
        super().__init__(parent)
        self.context_manager = context_manager
        self.collapsed = False
        self._init_ui()

    def _init_ui(self):
        self.setMinimumWidth(320)
        self.setMaximumWidth(480)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor("#181f2a"))
        self.setPalette(pal)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Header with collapse/expand
        header_layout = QHBoxLayout()
        self.title_label = QLabel("Workspace Context")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.title_label.setStyleSheet("font-weight: bold; font-size: 16px;")
        header_layout.addWidget(self.title_label)
        self.collapse_btn = QPushButton("-")
        self.collapse_btn.setFixedWidth(24)
        self.collapse_btn.setToolTip("Collapse/Expand panel")
        self.collapse_btn.clicked.connect(self.toggle_collapse)
        header_layout.addWidget(self.collapse_btn)
        self.layout.addLayout(header_layout)

        # State indicator
        self.state_label = QLabel("State: Idle")
        self.state_label.setStyleSheet("color: #22c55e; font-weight: bold;")
        self.state_label.setToolTip("Current workspace state")
        self.layout.addWidget(self.state_label)

        # Context display
        self.context_display = QTextEdit()
        self.context_display.setReadOnly(True)
        self.context_display.setToolTip("Current session context")
        self.layout.addWidget(self.context_display)

        # Progress
        progress_layout = QHBoxLayout()
        self.progress_label = QLabel("Progress:")
        self.progress_label.setToolTip("Current operation progress")
        progress_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setToolTip("Progress bar for current task")
        progress_layout.addWidget(self.progress_bar)
        self.layout.addLayout(progress_layout)

        # Live log area
        self.log_label = QLabel("Agent/Task Log:")
        self.log_label.setToolTip("Live log of agent and task events")
        self.layout.addWidget(self.log_label)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setMaximumHeight(120)
        self.log_area.setToolTip("Live log output. Auto-scroll enabled.")
        self.layout.addWidget(self.log_area)

        # Buttons
        btn_layout = QHBoxLayout()
        self.clear_btn = QPushButton("Clear Context")
        self.clear_btn.setToolTip("Clear all context and logs")
        self.clear_btn.clicked.connect(self.clear_context)
        btn_layout.addWidget(self.clear_btn)
        self.export_btn = QPushButton("Export Context")
        self.export_btn.setToolTip("Export context and logs to file")
        self.export_btn.clicked.connect(self.export_context)
        btn_layout.addWidget(self.export_btn)
        self.layout.addLayout(btn_layout)

        self.update_context_display()

    def update_context_display(self):
        context = self.context_manager.session_context
        text = "\n".join(f"{k}: {v}" for k, v in context.items())
        self.context_display.setText(text)
        self.context_updated.emit(context)

    def set_progress(self, value: int, text: str = None):
        self.progress_bar.setValue(value)
        if text:
            self.progress_label.setText(f"Progress: {text}")
        self.progress_updated.emit(value, text or "")

    def set_state(self, state: str):
        self.state_label.setText(f"State: {state}")
        # Visual indicator: green for active, gray for idle, red for error
        color = "#22c55e" if state.lower() == "active" else ("#ef4444" if state.lower() == "error" else "#94a3b8")
        self.state_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.state_changed.emit(state)

    def update_context(self, event):
        self.context_manager.update_context(event)
        self.update_context_display()

    def log(self, message: str):
        self.log_area.append(message)
        self.log_area.moveCursor(self.log_area.textCursor().End)

    def clear_context(self):
        self.context_manager.session_context.clear()
        self.context_display.clear()
        self.log_area.clear()
        self.set_progress(0, "")
        self.set_state("Idle")

    def export_context(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Export Context", "context.txt", "Text Files (*.txt)")
        if fname:
            with open(fname, "w") as f:
                f.write("[Context]\n")
                for k, v in self.context_manager.session_context.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n[Log]\n")
                f.write(self.log_area.toPlainText())

    def toggle_collapse(self):
        self.collapsed = not self.collapsed
        self.setVisible(not self.collapsed) 