from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox
from plugins.batch_processor import BatchProcessorPlugin

class BatchProcessingDialog(QDialog):
    def __init__(self, parent=None, agent=None):
        super().__init__(parent)
        self.agent = agent
        self.setWindowTitle("Batch Processing")
        layout = QVBoxLayout()
        self.input_label = QLabel("Input Directory:")
        self.input_edit = QLineEdit()
        self.input_btn = QPushButton("Browse")
        self.input_btn.clicked.connect(self.browse_input)
        self.output_label = QLabel("Output Directory:")
        self.output_edit = QLineEdit("output")
        self.output_btn = QPushButton("Browse")
        self.output_btn.clicked.connect(self.browse_output)
        self.run_btn = QPushButton("Run Batch")
        self.run_btn.clicked.connect(self.run_batch)
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_edit)
        layout.addWidget(self.input_btn)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_edit)
        layout.addWidget(self.output_btn)
        layout.addWidget(self.run_btn)
        self.setLayout(layout)

    def browse_input(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Input Directory")
        if dir:
            self.input_edit.setText(dir)

    def browse_output(self):
        dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir:
            self.output_edit.setText(dir)

    def run_batch(self):
        input_dir = self.input_edit.text()
        output_dir = self.output_edit.text()
        if not input_dir:
            QMessageBox.warning(self, "Error", "Please select input directory")
            return
        plugin = BatchProcessorPlugin()
        results = plugin.run(input_dir, self.agent, output_dir)
        QMessageBox.information(self, "Results", str(results))
