"""
Model Versioning System
Handles semantic versioning and model lineage tracking
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import semver
import json
from pathlib import Path


@dataclass
class ModelLineage:
    """Tracks the evolution and relationships between model versions"""

    parent_version: Optional[str]
    derived_versions: List[str]
    training_data: str
    training_config: Dict[str, Any]
    creation_date: datetime
    author: str
    commit_hash: Optional[str]
    description: str


class ModelVersion:
    """Represents a specific version of a model with semantic versioning"""

    def __init__(
        self,
        model_id: str,
        version: str,
        hash: str,
        url: str,
        metadata: Dict[str, Any],
        lineage: Optional[ModelLineage] = None,
    ):
        self.model_id = model_id
        self._version = semver.VersionInfo.parse(version)
        self.hash = hash
        self.url = url
        self.metadata = metadata
        self.lineage = lineage
        self.download_date = None
        self.last_validated = None
        self.performance_metrics = {}

    @property
    def version(self) -> str:
        return str(self._version)

    def bump_major(self) -> "ModelVersion":
        """Create new version with major version bump"""
        new_version = self._version.bump_major()
        return self._create_new_version(new_version)

    def bump_minor(self) -> "ModelVersion":
        """Create new version with minor version bump"""
        new_version = self._version.bump_minor()
        return self._create_new_version(new_version)

    def bump_patch(self) -> "ModelVersion":
        """Create new version with patch version bump"""
        new_version = self._version.bump_patch()
        return self._create_new_version(new_version)

    def _create_new_version(self, new_version: semver.VersionInfo) -> "ModelVersion":
        """Helper to create new version instance"""
        return ModelVersion(
            model_id=self.model_id,
            version=str(new_version),
            hash=self.hash,
            url=self.url,
            metadata=self.metadata.copy(),
            lineage=ModelLineage(
                parent_version=self.version,
                derived_versions=[],
                training_data=self.lineage.training_data if self.lineage else "",
                training_config=self.lineage.training_config.copy() if self.lineage else {},
                creation_date=datetime.now(),
                author=self.lineage.author if self.lineage else "",
                commit_hash=None,
                description="",
            ),
        )


class VersionManager:
    """Manages model versions and their relationships"""

    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.storage_dir / "versions.json"
        self.versions: Dict[str, Dict[str, ModelVersion]] = {}
        self._load_versions()

    def _load_versions(self) -> None:
        """Load version information from disk"""
        if not self.versions_file.exists():
            return

        try:
            with open(self.versions_file, "r") as f:
                data = json.load(f)
                for model_id, versions in data.items():
                    self.versions[model_id] = {}
                    for v in versions:
                        lineage = ModelLineage(**v["lineage"]) if v.get("lineage") else None
                        version = ModelVersion(
                            model_id=model_id,
                            version=v["version"],
                            hash=v["hash"],
                            url=v["url"],
                            metadata=v["metadata"],
                            lineage=lineage,
                        )
                        version.download_date = v.get("download_date")
                        version.last_validated = v.get("last_validated")
                        version.performance_metrics = v.get("performance_metrics", {})
                        self.versions[model_id][version.version] = version
        except Exception as e:
            print(f"Error loading versions: {e}")
            self.versions = {}

    def _save_versions(self) -> None:
        """Save version information to disk"""
        data = {}
        for model_id, versions in self.versions.items():
            data[model_id] = []
            for version in versions.values():
                version_data = {
                    "version": version.version,
                    "hash": version.hash,
                    "url": version.url,
                    "metadata": version.metadata,
                    "download_date": version.download_date,
                    "last_validated": version.last_validated,
                    "performance_metrics": version.performance_metrics,
                }
                if version.lineage:
                    version_data["lineage"] = {
                        "parent_version": version.lineage.parent_version,
                        "derived_versions": version.lineage.derived_versions,
                        "training_data": version.lineage.training_data,
                        "training_config": version.lineage.training_config,
                        "creation_date": version.lineage.creation_date.isoformat(),
                        "author": version.lineage.author,
                        "commit_hash": version.lineage.commit_hash,
                        "description": version.lineage.description,
                    }
                data[model_id].append(version_data)

        with open(self.versions_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_version(
        self,
        model_id: str,
        version: str,
        hash: str,
        url: str,
        metadata: Dict[str, Any],
        lineage: Optional[ModelLineage] = None,
    ) -> ModelVersion:
        """Register a new model version"""
        if model_id not in self.versions:
            self.versions[model_id] = {}

        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            hash=hash,
            url=url,
            metadata=metadata,
            lineage=lineage,
        )

        self.versions[model_id][version] = model_version
        self._save_versions()
        return model_version

    def get_version(self, model_id: str, version: str) -> Optional[ModelVersion]:
        """Get a specific model version"""
        return self.versions.get(model_id, {}).get(version)

    def get_latest_version(self, model_id: str) -> Optional[ModelVersion]:
        """Get the latest version of a model"""
        if model_id not in self.versions or not self.versions[model_id]:
            return None

        latest = max(
            self.versions[model_id].values(), key=lambda v: semver.VersionInfo.parse(v.version)
        )
        return latest

    def get_version_history(self, model_id: str) -> List[ModelVersion]:
        """Get version history for a model"""
        if model_id not in self.versions:
            return []

        versions = list(self.versions[model_id].values())
        versions.sort(key=lambda v: semver.VersionInfo.parse(v.version))
        return versions

    def get_lineage_tree(self, model_id: str) -> Dict[str, List[str]]:
        """Get the version lineage tree for a model"""
        if model_id not in self.versions:
            return {}

        tree = {}
        for version in self.versions[model_id].values():
            if version.lineage:
                parent = version.lineage.parent_version or "root"
                if parent not in tree:
                    tree[parent] = []
                tree[parent].append(version.version)
        return tree
