"""
Security System
Handles authentication, encryption, and API key management
"""

import os
import json
import base64
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger


class SecurityManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.keys_file = self.config_dir / "api_keys.enc"
        self.auth_file = self.config_dir / "auth.json"
        self.salt_file = self.config_dir / "salt"

        # Initialize encryption key
        self._init_encryption()

        # Load or create salt
        if not self.salt_file.exists():
            self.salt = os.urandom(16)
            with open(self.salt_file, "wb") as f:
                f.write(self.salt)
        else:
            with open(self.salt_file, "rb") as f:
                self.salt = f.read()

        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = self._load_api_keys()

    def _init_encryption(self) -> None:
        """Initialize or load encryption key"""
        key_file = self.config_dir / "key.enc"
        if not key_file.exists():
            self.encryption_key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(self.encryption_key)
        else:
            with open(key_file, "rb") as f:
                self.encryption_key = f.read()

        self.fernet = Fernet(self.encryption_key)

    def _hash_password(self, password: str) -> str:
        """Hash password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.b64encode(kdf.derive(password.encode()))
        return key.decode()

    def create_user(self, username: str, password: str) -> bool:
        """Create a new user"""
        try:
            if self.auth_file.exists():
                with open(self.auth_file, "r") as f:
                    users = json.load(f)
            else:
                users = {}

            if username in users:
                return False

            users[username] = {
                "password_hash": self._hash_password(password),
                "created_at": datetime.now().isoformat(),
                "last_login": None,
            }

            with open(self.auth_file, "w") as f:
                json.dump(users, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False

    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return session token"""
        try:
            if not self.auth_file.exists():
                return None

            with open(self.auth_file, "r") as f:
                users = json.load(f)

            if username not in users:
                return None

            if users[username]["password_hash"] != self._hash_password(password):
                return None

            # Create session
            session_token = base64.b64encode(os.urandom(32)).decode()
            self.sessions[session_token] = {
                "username": username,
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            }

            # Update last login
            users[username]["last_login"] = datetime.now().isoformat()
            with open(self.auth_file, "w") as f:
                json.dump(users, f, indent=2)

            return session_token

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return None

    def validate_session(self, session_token: str) -> bool:
        """Validate session token"""
        if session_token not in self.sessions:
            return False

        session = self.sessions[session_token]
        expires_at = datetime.fromisoformat(session["expires_at"])

        if datetime.now() > expires_at:
            del self.sessions[session_token]
            return False

        return True

    def _load_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Load encrypted API keys"""
        if not self.keys_file.exists():
            return {}

        try:
            with open(self.keys_file, "rb") as f:
                encrypted_data = f.read()
            decrypted_data = self.fernet.decrypt(encrypted_data)
            return json.loads(decrypted_data)
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            return {}

    def _save_api_keys(self) -> None:
        """Save encrypted API keys"""
        try:
            encrypted_data = self.fernet.encrypt(json.dumps(self.api_keys).encode())
            with open(self.keys_file, "wb") as f:
                f.write(encrypted_data)
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")

    def add_api_key(self, service: str, key: str, metadata: Dict[str, Any] = None) -> bool:
        """Add or update API key"""
        try:
            self.api_keys[service] = {
                "key": key,
                "added_at": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
            self._save_api_keys()
            return True
        except Exception as e:
            logger.error(f"Failed to add API key: {e}")
            return False

    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for service"""
        try:
            return self.api_keys.get(service, {}).get("key")
        except Exception as e:
            logger.error(f"Failed to get API key: {e}")
            return None

    def remove_api_key(self, service: str) -> bool:
        """Remove API key"""
        try:
            if service in self.api_keys:
                del self.api_keys[service]
                self._save_api_keys()
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove API key: {e}")
            return False

    def encrypt_data(self, data: str) -> Optional[str]:
        """Encrypt sensitive data"""
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}")
            return None

    def decrypt_data(self, encrypted_data: str) -> Optional[str]:
        """Decrypt sensitive data"""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}")
            return None

    def sandbox_plugin(self, plugin_path: str) -> bool:
        # TODO: Run plugin in a secure sandbox
        logger.warning("Plugin sandboxing not yet implemented.")
        return False

    def check_permissions(self, user_id: str, action: str) -> bool:
        # TODO: Check if user has permission for action
        logger.warning("Permission checking not yet implemented.")
        return True

    # Unlimited error handling and automatic recovery
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """Handle errors with unlimited retries and automatic recovery."""
        from src.core.error_recovery import ErrorRecovery

        recovery = ErrorRecovery()
        retries = 0
        max_retries = 1000  # Unlimited for practical purposes
        while retries < max_retries:
            if recovery.recover(error, context or {}):
                retries += 1
                continue
            else:
                break
        logger.error(f"Error could not be recovered after {retries} attempts: {error}")
        return False

    def report_crash(self, error: Exception, context: Dict[str, Any] = None) -> str:
        """Report a crash using the error recovery system."""
        from src.core.error_recovery import ErrorRecovery

        recovery = ErrorRecovery()
        return recovery._save_crash_report(error, context or {})

    def log_event(self, message: str, level: str = "INFO") -> None:
        """Log an event with the specified level."""
        if level == "DEBUG":
            logger.debug(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        else:
            logger.info(message)
