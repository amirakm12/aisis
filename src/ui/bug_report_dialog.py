from PySide6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox

class BugReportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Report Bug")
        layout = QVBoxLayout()
        self.description = QTextEdit()
        self.description.setPlaceholderText("Describe the bug...")
        layout.addWidget(self.description)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.setLayout(layout)
