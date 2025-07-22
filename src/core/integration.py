from typing import Any, Dict


class BaseIntegration:
    """
    Abstract base class for external service integrations (cloud, collaboration,
    etc.).
    """

    name: str = "base"

    def connect(self, credentials: Dict[str, Any]) -> bool:
        """Connect to the service using provided credentials."""
        raise NotImplementedError

    def upload(self, file_path: str, dest_path: str) -> bool:
        """Upload a file to the service."""
        raise NotImplementedError

    def download(self, remote_path: str, local_path: str) -> bool:
        """Download a file from the service."""
        raise NotImplementedError

    def list_files(self, folder_path: str) -> list:
        """List files in a folder on the service."""
        raise NotImplementedError


# Example stub for Google Drive integration
class GoogleDriveIntegration(BaseIntegration):
    name = "google_drive"

    def connect(self, credentials: Dict[str, Any]) -> bool:
        # TODO: Implement Google Drive OAuth2 connection
        return True

    def upload(self, file_path: str, dest_path: str) -> bool:
        # TODO: Implement file upload
        return True

    def download(self, remote_path: str, local_path: str) -> bool:
        # TODO: Implement file download
        return True

    def list_files(self, folder_path: str) -> list:
        # TODO: Implement file listing
        return []


# Example stub for Dropbox integration
class DropboxIntegration(BaseIntegration):
    name = "dropbox"

    def connect(self, credentials: Dict[str, Any]) -> bool:
        # TODO: Implement Dropbox OAuth2 connection
        return True

    def upload(self, file_path: str, dest_path: str) -> bool:
        # TODO: Implement file upload
        return True

    def download(self, remote_path: str, local_path: str) -> bool:
        # TODO: Implement file download
        return True

    def list_files(self, folder_path: str) -> list:
        # TODO: Implement file listing
        return []


# Integration registry for easy discovery and management
INTEGRATION_REGISTRY = {
    GoogleDriveIntegration.name: GoogleDriveIntegration(),
    DropboxIntegration.name: DropboxIntegration(),
}


"""
How to add a new integration:
- Subclass BaseIntegration and implement required methods.
- Add an instance to INTEGRATION_REGISTRY.

This structure supports future plugin/extension loading for third-party
integrations.
"""
