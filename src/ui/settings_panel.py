from PySide6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QFormLayout, QLineEdit, QCheckBox, QPushButton, QFileDialog, QMessageBox, QComboBox
import json

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["dark", "light"])
        self.theme_combo.setCurrentText(config.get("ui.theme", "dark"))
        self.model_dir_box = QLineEdit(str(config.get("paths.models_dir", "models")))
        self.gpu_checkbox = QCheckBox("Enable GPU")
        self.gpu_checkbox.setChecked(config.get("gpu.use_cuda", True))
        general_layout.addRow("Theme:", self.theme_combo)
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
        self.config.set("ui.theme", self.theme_combo.currentText())
        self.config.set("gpu.use_cuda", self.gpu_checkbox.isChecked())
        self.config.set("paths.models_dir", self.model_dir_box.text())

        # Apply theme immediately
        from PySide6.QtWidgets import QApplication
        from .theme_manager import ThemeManager
        app = QApplication.instance()
        if self.config.get("ui.theme") == "dark":
            ThemeManager.apply_dark(app)
        else:
            ThemeManager.apply_light(app)

        QMessageBox.information(self, "Save", "Settings saved successfully.")

    def export_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Settings", "settings.json", "JSON Files (*.json)")
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self.config.data, f, indent=2)
                QMessageBox.information(self, "Export", "Settings exported successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def import_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Settings", "", "JSON Files (*.json)")
        if path:
            try:
                with open(path, "r") as f:
                    imported = json.load(f)
                self.config.data = imported
                self.config.save()
                QMessageBox.information(self, "Import", "Settings imported. Some changes may require restart.")
            except Exception as e:
                QMessageBox.critical(self, "Import Error", str(e))
