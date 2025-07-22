"""
Model Management System
Handles model versioning, downloads, validation, and performance benchmarking
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
from datetime import datetime

try:
    import torch
    import requests
    from tqdm import tqdm
    from loguru import logger

    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False
    if TYPE_CHECKING:
        import torch
        import requests
        from tqdm import tqdm
        from loguru import logger

from .error_recovery import error_recovery


class ModelVersion:
    def __init__(self, version: str, hash: str, url: str, metadata: Dict[str, Any]):
        self.version = version
        self.hash = hash
        self.url = url
        self.metadata = metadata
        self.download_date: Optional[float] = None
        self.last_validated: Optional[float] = None
        self.performance_metrics: Dict[str, Any] = {}


class ModelManager:
    def __init__(self, models_dir: str = "models"):
        if not HAS_DEPENDENCIES:
            raise ImportError(
                "Required dependencies not found. Please install: torch, requests, tqdm, loguru"
            )

        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.versions_file = self.models_dir / "versions.json"
        self.versions: Dict[str, Dict[str, ModelVersion]] = {}
        self.load_versions()

        # Register model-specific error handlers
        error_recovery.register_recovery_handler("ModelDownloadError", self._handle_download_error)
        error_recovery.register_recovery_handler(
            "ModelValidationError", self._handle_validation_error
        )
        error_recovery.register_recovery_handler("ModelLoadError", self._handle_load_error)

    def _handle_download_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle model download errors"""
        model_id = context.get("model_id")
        version = context.get("version")
        if not model_id or not version:
            return False

        try:
            # Clean up partial downloads
            model_path = self.models_dir / f"{model_id}-{version}"
            if model_path.exists():
                os.remove(model_path)

            # Try alternative download URL if available
            model_version = self.versions[model_id][version]
            if "backup_url" in model_version.metadata:
                model_version.url = model_version.metadata["backup_url"]
                return True

            return False
        except Exception:
            return False

    def _handle_validation_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle model validation errors"""
        model_id = context.get("model_id")
        version = context.get("version")
        if not model_id or not version:
            return False

        try:
            # Clean up invalid model file
            model_path = self.models_dir / f"{model_id}-{version}"
            if model_path.exists():
                os.remove(model_path)
            return True
        except Exception:
            return False

    def _handle_load_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle model loading errors"""
        if isinstance(error, torch.cuda.OutOfMemoryError):
            try:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Try CPU fallback
                context["device"] = "cpu"
                return True
            except Exception:
                return False
        return False

    @error_recovery.with_recovery({"module": "model_manager"})
    def load_versions(self) -> None:
        """Load model versions from versions.json"""
        if self.versions_file.exists():
            with open(self.versions_file, "r") as f:
                data = json.load(f)
                for model_id, versions in data.items():
                    self.versions[model_id] = {
                        v["version"]: ModelVersion(
                            version=v["version"],
                            hash=v["hash"],
                            url=v["url"],
                            metadata=v["metadata"],
                        )
                        for v in versions
                    }

    @error_recovery.with_recovery({"module": "model_manager"})
    def save_versions(self) -> None:
        """Save model versions to versions.json"""
        data = {
            model_id: [
                {"version": v.version, "hash": v.hash, "url": v.url, "metadata": v.metadata}
                for v in versions.values()
            ]
            for model_id, versions in self.versions.items()
        }
        with open(self.versions_file, "w") as f:
            json.dump(data, f, indent=2)

    @error_recovery.with_recovery()
    async def download_model(self, model_id: str, version: str) -> bool:
        """Download a specific model version"""
        if model_id not in self.versions or version not in self.versions[model_id]:
            raise ValueError(f"Model {model_id} version {version} not found")

        model_version = self.versions[model_id][version]
        model_path = self.models_dir / f"{model_id}-{version}"

        try:
            # Download with progress bar
            response = requests.get(model_version.url, stream=True)
            total_size = int(response.headers.get("content-length", 0))

            with (
                open(model_path, "wb") as f,
                tqdm(
                    desc=f"Downloading {model_id} v{version}",
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                ) as pbar,
            ):
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)

            # Validate download
            if await self.validate_model(model_id, version):
                model_version.download_date = time.time()
                self.save_versions()
                return True
            else:
                os.remove(model_path)
                return False

        except Exception as e:
            logger.error(f"Failed to download model {model_id} v{version}: {e}")
            if model_path.exists():
                os.remove(model_path)
            raise ModelDownloadError(str(e))

    @error_recovery.with_recovery()
    async def validate_model(self, model_id: str, version: str) -> bool:
        """Validate model file integrity"""
        if model_id not in self.versions or version not in self.versions[model_id]:
            raise ValueError(f"Model {model_id} version {version} not found")

        model_version = self.versions[model_id][version]
        model_path = self.models_dir / f"{model_id}-{version}"

        if not model_path.exists():
            return False

        try:
            # Calculate file hash
            sha256_hash = hashlib.sha256()
            with open(model_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            # Verify hash
            if sha256_hash.hexdigest() == model_version.hash:
                model_version.last_validated = time.time()
                self.save_versions()
                return True
            raise ModelValidationError("Model file hash mismatch")

        except Exception as e:
            logger.error(f"Failed to validate model {model_id} v{version}: {e}")
            raise ModelValidationError(str(e))

    @error_recovery.with_recovery()
    async def benchmark_model(self, model_id: str, version: str) -> Dict[str, float]:
        """Run performance benchmarks on a model"""
        if model_id not in self.versions or version not in self.versions[model_id]:
            raise ValueError(f"Model {model_id} version {version} not found")

        model_version = self.versions[model_id][version]
        model_path = self.models_dir / f"{model_id}-{version}"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Load model
            model = torch.load(model_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()

            # Run benchmarks
            metrics = {}

            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    # Example inference
                    input_tensor = torch.randn(1, 3, 224, 224).to(device)
                    _ = model(input_tensor)
            inference_time = (time.time() - start_time) / 100

            metrics["avg_inference_time"] = inference_time
            metrics["device"] = str(device)
            metrics["timestamp"] = time.time()

            # Save metrics
            model_version.performance_metrics = metrics
            self.save_versions()

            return metrics

        except Exception as e:
            logger.error(f"Failed to benchmark model {model_id} v{version}: {e}")
            raise ModelLoadError(str(e))

    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available models and their versions"""
        return {model_id: list(versions.keys()) for model_id, versions in self.versions.items()}

    def get_model_info(self, model_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model version"""
        if model_id not in self.versions or version not in self.versions[model_id]:
            return None

        model_version = self.versions[model_id][version]
        return {
            "version": model_version.version,
            "hash": model_version.hash,
            "metadata": model_version.metadata,
            "download_date": model_version.download_date,
            "last_validated": model_version.last_validated,
            "performance_metrics": model_version.performance_metrics,
        }

    def register_model(
        self, model_id: str, version: str, url: str, hash: str, metadata: Dict[str, Any]
    ) -> None:
        """Register a new model version"""
        if model_id not in self.versions:
            self.versions[model_id] = {}

        self.versions[model_id][version] = ModelVersion(
            version=version, hash=hash, url=url, metadata=metadata
        )
        self.save_versions()


# Custom exceptions
class ModelDownloadError(Exception):
    pass


class ModelValidationError(Exception):
    pass


class ModelLoadError(Exception):
    pass
