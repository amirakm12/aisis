from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QLabel, QPushButton, QTextEdit, QHBoxLayout, QProgressBar

class ModelZooDialog(QDialog):
    """
    Dialog for browsing, downloading, and switching models from the Model Zoo.
    """
    def __init__(self, model_zoo, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Zoo")
        layout = QVBoxLayout(self)
        self.model_list = QListWidget()
        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.download_btn = QPushButton("Download")
        self.activate_btn = QPushButton("Activate")
        self.progress = QProgressBar()
        layout.addWidget(QLabel("Available Models:"))
        layout.addWidget(self.model_list)
        layout.addWidget(QLabel("Model Details:"))
        layout.addWidget(self.details)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.download_btn)
        button_layout.addWidget(self.activate_btn)
        layout.addLayout(button_layout)
        layout.addWidget(self.progress)
        self.setLayout(layout)
        self.model_zoo = model_zoo
        self.populate_models()
        self.model_list.currentTextChanged.connect(self.show_details)
        # TODO: Connect download/activate buttons to backend logic

    def populate_models(self):
        self.model_list.clear()
        for model in self.model_zoo.list_models():
            self.model_list.addItem(model["name"])

    def show_details(self, model_name):
        model = next((m for m in self.model_zoo.list_models() if m["name"] == model_name), None)
        if model:
            self.details.setText(f"Type: {model['type']}\nVersion: {model['version']}\nStatus: {model['status']}")
        else:
            self.details.setText("No details available.") 