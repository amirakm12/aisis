from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QFileDialog,
    QTabWidget,
    QWidget,
    QHBoxLayout,
)


class AgentInvocationDialog(QDialog):
    """
    Dialog for providing input (image, text, sketch) to an agent for invocation.
    Supports both automated and manual agent invocation.
    """

    def __init__(self, agent_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Invoke Agent: {agent_name}")
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        # Image input tab
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        self.image_path = QLineEdit()
        self.browse_button = QPushButton("Browse Image")
        image_layout.addWidget(QLabel("Image Path:"))
        image_layout.addWidget(self.image_path)
        image_layout.addWidget(self.browse_button)
        self.browse_button.clicked.connect(self.browse_image)
        self.tabs.addTab(image_tab, "Image")
        # Text input tab
        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        self.text_input = QTextEdit()
        text_layout.addWidget(QLabel("Text Input:"))
        text_layout.addWidget(self.text_input)
        self.tabs.addTab(text_tab, "Text")
        # Sketch input tab (stub)
        sketch_tab = QWidget()
        sketch_layout = QVBoxLayout(sketch_tab)
        sketch_layout.addWidget(
            QLabel("Sketch Input: (integrate with DrawingCanvas)")
        )
        self.tabs.addTab(sketch_tab, "Sketch")
        layout.addWidget(self.tabs)
        self.submit_button = QPushButton("Submit")
        layout.addWidget(self.submit_button)
        # TODO: Connect submit_button to agent invocation logic

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image_path.setText(file_path)
