"""
Security Dialog
Handles user authentication and API key management UI
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTabWidget,
    QWidget, QFormLayout, QMessageBox,
    QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt
from typing import Optional, Dict, Any

from src.core.security import SecurityManager

class LoginDialog(QDialog):
    def __init__(self, security_manager: SecurityManager, parent=None):
        super().__init__(parent)
        self.security_manager = security_manager
        self.session_token = None
        self._init_ui()

    def _init_ui(self):
        """Initialize the login dialog UI"""
        self.setWindowTitle("Login")
        self.setModal(True)

        layout = QVBoxLayout()

        # Username field
        username_layout = QHBoxLayout()
        username_label = QLabel("Username:")
        self.username_input = QLineEdit()
        username_layout.addWidget(username_label)
        username_layout.addWidget(self.username_input)
        layout.addLayout(username_layout)

        # Password field
        password_layout = QHBoxLayout()
        password_label = QLabel("Password:")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        password_layout.addWidget(password_label)
        password_layout.addWidget(self.password_input)
        layout.addLayout(password_layout)

        # Buttons
        button_layout = QHBoxLayout()
        login_button = QPushButton("Login")
        login_button.clicked.connect(self._handle_login)
        register_button = QPushButton("Register")
        register_button.clicked.connect(self._handle_register)
        button_layout.addWidget(login_button)
        button_layout.addWidget(register_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _handle_login(self):
        """Handle login button click"""
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(
                self,
                "Error",
                "Please enter both username and password"
            )
            return

        session_token = self.security_manager.authenticate(username, password)
        if session_token:
            self.session_token = session_token
            self.accept()
        else:
            QMessageBox.warning(
                self,
                "Error",
                "Invalid username or password"
            )

    def _handle_register(self):
        """Handle register button click"""
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(
                self,
                "Error",
                "Please enter both username and password"
            )
            return

        if self.security_manager.create_user(username, password):
            QMessageBox.information(
                self,
                "Success",
                "User registered successfully. You can now log in."
            )
        else:
            QMessageBox.warning(
                self,
                "Error",
                "Username already exists"
            )

class APIKeyDialog(QDialog):
    def __init__(self, security_manager: SecurityManager, parent=None):
        super().__init__(parent)
        self.security_manager = security_manager
        self._init_ui()

    def _init_ui(self):
        """Initialize the API key management dialog UI"""
        self.setWindowTitle("API Key Management")
        self.setModal(True)
        self.resize(600, 400)

        layout = QVBoxLayout()

        # API Key Table
        self.api_key_table = QTableWidget()
        self.api_key_table.setColumnCount(3)
        self.api_key_table.setHorizontalHeaderLabels(["Service", "Key", "Added"])
        layout.addWidget(self.api_key_table)

        # Add API Key Form
        form_layout = QFormLayout()
        self.service_input = QLineEdit()
        self.key_input = QLineEdit()
        form_layout.addRow("Service:", self.service_input)
        form_layout.addRow("API Key:", self.key_input)
        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()
        add_button = QPushButton("Add Key")
        add_button.clicked.connect(self._handle_add_key)
        remove_button = QPushButton("Remove Selected")
        remove_button.clicked.connect(self._handle_remove_key)
        button_layout.addWidget(add_button)
        button_layout.addWidget(remove_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)
        self._load_api_keys()

    def _load_api_keys(self):
        """Load and display API keys"""
        self.api_key_table.setRowCount(0)
        for service, data in self.security_manager.api_keys.items():
            row = self.api_key_table.rowCount()
            self.api_key_table.insertRow(row)
            self.api_key_table.setItem(row, 0, QTableWidgetItem(service))
            # Show only first/last 4 chars of key
            key = data["key"]
            masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            self.api_key_table.setItem(row, 1, QTableWidgetItem(masked_key))
            self.api_key_table.setItem(row, 2, QTableWidgetItem(data["added_at"]))

    def _handle_add_key(self):
        """Handle add API key button click"""
        service = self.service_input.text()
        key = self.key_input.text()

        if not service or not key:
            QMessageBox.warning(
                self,
                "Error",
                "Please enter both service name and API key"
            )
            return

        if self.security_manager.add_api_key(service, key):
            self.service_input.clear()
            self.key_input.clear()
            self._load_api_keys()
            QMessageBox.information(
                self,
                "Success",
                f"API key for {service} added successfully"
            )
        else:
            QMessageBox.warning(
                self,
                "Error",
                "Failed to add API key"
            )

    def _handle_remove_key(self):
        """Handle remove API key button click"""
        selected_items = self.api_key_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self,
                "Error",
                "Please select an API key to remove"
            )
            return

        service = self.api_key_table.item(selected_items[0].row(), 0).text()
        if self.security_manager.remove_api_key(service):
            self._load_api_keys()
            QMessageBox.information(
                self,
                "Success",
                f"API key for {service} removed successfully"
            )
        else:
            QMessageBox.warning(
                self,
                "Error",
                "Failed to remove API key"
            )

class SecurityDialog(QDialog):
    def __init__(self, security_manager: SecurityManager, parent=None):
        super().__init__(parent)
        self.security_manager = security_manager
        self._init_ui()

    def _init_ui(self):
        """Initialize the main security dialog UI"""
        self.setWindowTitle("Security Settings")
        self.setModal(True)
        self.resize(800, 600)

        layout = QVBoxLayout()

        # Tab widget
        tabs = QTabWidget()
        
        # API Keys tab
        api_keys_tab = APIKeyDialog(self.security_manager)
        tabs.addTab(api_keys_tab, "API Keys")

        # Add tabs widget to layout
        layout.addWidget(tabs)

        # Status bar
        self.status_label = QLabel()
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    def show_login(self) -> Optional[str]:
        """Show login dialog and return session token"""
        login_dialog = LoginDialog(self.security_manager, self)
        if login_dialog.exec() == QDialog.DialogCode.Accepted:
            return login_dialog.session_token
        return None 