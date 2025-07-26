from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QPushButton, QHBoxLayout

class PluginManagerDialog(QDialog):
    def __init__(self, plugin_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plugin Manager")
        layout = QVBoxLayout(self)
        self.plugin_list = QListWidget()
        self.enable_btn = QPushButton("Enable")
        self.disable_btn = QPushButton("Disable")
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.enable_btn)
        btn_layout.addWidget(self.disable_btn)
        layout.addWidget(self.plugin_list)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        # TODO: Populate and connect logic as needed 