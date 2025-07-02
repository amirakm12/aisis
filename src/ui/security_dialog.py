from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QListWidget

class SecurityDialog(QDialog):
    """
    Dialog for user authentication, plugin sandboxing, and permission management.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Security & Permissions")
        layout = QVBoxLayout(self)
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("User ID")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.login_button = QPushButton("Login")
        self.sandbox_checkbox = QCheckBox("Enable Plugin Sandbox")
        self.permission_list = QListWidget()
        layout.addWidget(QLabel("User Authentication:"))
        layout.addWidget(self.user_input)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)
        layout.addWidget(self.sandbox_checkbox)
        layout.addWidget(QLabel("Permissions:"))
        layout.addWidget(self.permission_list)
        # TODO: Connect UI to SecurityManager 