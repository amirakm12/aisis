from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QListWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QHBoxLayout,
)


class ModelZooDialog(QDialog):
    """
    Dialog for browsing, downloading, and switching models from the Model Zoo.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Zoo")
        layout = QVBoxLayout(self)
        self.model_list = QListWidget()
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.download_button = QPushButton("Download")
        self.switch_button = QPushButton("Switch")
        layout.addWidget(QLabel("Available Models:"))
        layout.addWidget(self.model_list)
        layout.addWidget(QLabel("Model Details:"))
        layout.addWidget(self.details)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.download_button)
        button_layout.addWidget(self.switch_button)
        layout.addLayout(button_layout)
        # TODO: Connect buttons to ModelZoo backend
