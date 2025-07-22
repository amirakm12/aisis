from PySide6.QtWidgets import QWizard, QWizardPage, QLabel, QVBoxLayout, QCheckBox, QComboBox, QPushButton

from ..core.config import config

class SetupWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AISIS First-Time Setup Wizard")
        self.setWizardStyle(QWizard.ModernStyle)
        self.addPage(IntroPage(self))
        self.addPage(ConfigPage(self))
        self.addPage(ModelPage(self))
        self.addPage(FinishPage(self))

class IntroPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome to AISIS")
        label = QLabel("This wizard will help you set up AISIS for the first time.")
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)

class ConfigPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Configuration")
        label = QLabel("Select your preferences:")
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.theme_combo)
        self.setLayout(layout)

    def initializePage(self):
        self.theme_combo.setCurrentText(config.get("theme", "Dark"))

class ModelPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Download Models")
        label = QLabel("Select models to download:")
        self.whisper_check = QCheckBox("Whisper ASR")
        self.bark_check = QCheckBox("Bark TTS")
        self.whisper_check.setChecked(True)
        self.bark_check.setChecked(True)
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.whisper_check)
        layout.addWidget(self.bark_check)
        self.setLayout(layout)

class FinishPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Setup Complete")
        label = QLabel("AISIS is now set up. Click Finish to start.")
        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)

    def initializePage(self):
        # Save configs
        theme = self.wizard().page(1).theme_combo.currentText()
        config["theme"] = theme.lower()
        # Trigger model downloads based on checks
        if self.wizard().page(2).whisper_check.isChecked():
            print("Downloading Whisper...")  # Call download function
        if self.wizard().page(2).bark_check.isChecked():
            print("Downloading Bark...")  # Call download function

