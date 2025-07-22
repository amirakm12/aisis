from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTabWidget,
    QWidget,
    QFormLayout,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
)
import json


class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        self.theme_box = QLineEdit(config.get("ui.theme", "dark"))
        self.model_dir_box = QLineEdit(str(config.get("paths.models_dir", "models")))
        self.gpu_checkbox = QCheckBox("Enable GPU")
        self.gpu_checkbox.setChecked(config.get("gpu.use_cuda", True))
        general_layout.addRow("Theme:", self.theme_box)
        general_layout.addRow("Model Directory:", self.model_dir_box)
        general_layout.addRow(self.gpu_checkbox)
        tabs.addTab(general_tab, "General")
        layout.addWidget(tabs)
        self.save_btn = QPushButton("Save")
        self.export_btn = QPushButton("Export Settings")
        self.import_btn = QPushButton("Import Settings")
        layout.addWidget(self.save_btn)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.import_btn)
        self.setLayout(layout)
        self.save_btn.clicked.connect(self.save_settings)
        self.export_btn.clicked.connect(self.export_settings)
        self.import_btn.clicked.connect(self.import_settings)
        self.config = config

    def save_settings(self):
        # TODO: Save settings to config
        pass

    def export_settings(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Settings", "settings.json", "JSON Files (*.json)"
        )
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self.config, f, indent=2)
                QMessageBox.information(self, "Export", "Settings exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def import_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Settings", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, "r") as f:
                    imported = json.load(f)
                # TODO: Validate and apply imported settings
                QMessageBox.information(
                    self, "Import", "Settings imported. Please restart the app."
                )
            except Exception as e:
                QMessageBox.critical(self, "Import Error", str(e))
        # TODO: Connect save_btn to config update logic
