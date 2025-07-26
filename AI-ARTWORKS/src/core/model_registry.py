"""
Central Model Registry and Configuration System
Manages model registration, configuration, and discovery
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from enum import Enum


class ModelType(Enum):
    """Types of AI models"""
    LLM = "llm"  # Language models
    VISION = "vision"  # Computer vision models
    AUDIO = "audio"  # Audio processing models
    MULTIMODAL = "multimodal"  # Multi-modal models
    DIFFUSION = "diffusion"  # Diffusion models
    RESTORATION = "restoration"  # Image restoration models
    STYLE = "style"  # Style transfer models
    SEMANTIC = "semantic"  # Semantic segmentation models
    RETOUCH = "retouch"  # Photo retouching models
    NERF = "nerf"  # Neural radiance fields


class ModelProvider(Enum):
    """Model providers/sources"""
    HUGGINGFACE = "huggingface"
    PYTORCH = "pytorch"
    ONNX = "onnx"
    CUSTOM = "custom"
    LOCAL = "local"


@dataclass
class ModelConfig:
    """Configuration for a registered model"""
    name: str
    model_type: ModelType
    provider: ModelProvider
    description: str
    version: str
    url: str
    hash: str
    size: int
    requirements: Dict[str, str]
    capabilities: List[str]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    is_downloaded: bool = False
    last_validated: Optional[datetime] = None
    last_benchmarked: Optional[datetime] = None


class ModelRegistry:
    """Central registry for managing AI models"""

    def __init__(self, registry_dir: Path):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.registry_dir / "registry.json"
        self.models: Dict[str, ModelConfig] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load model registry from disk"""
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, "r") as f:
                data = json.load(f)
                for model_id, config in data.items():
                    self.models[model_id] = ModelConfig(
                        name=config["name"],
                        model_type=ModelType(config["model_type"]),
                        provider=ModelProvider(config["provider"]),
                        description=config["description"],
                        version=config["version"],
                        url=config["url"],
                        hash=config["hash"],
                        size=config["size"],
                        requirements=config["requirements"],
                        capabilities=config["capabilities"],
                        parameters=config["parameters"],
                        metadata=config["metadata"],
                        is_downloaded=config["is_downloaded"],
                        last_validated=datetime.fromisoformat(config["last_validated"])
                        if config.get("last_validated")
                        else None,
                        last_benchmarked=datetime.fromisoformat(config["last_benchmarked"])
                        if config.get("last_benchmarked")
                        else None
                    )
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            self.models = {}

    def _save_registry(self) -> None:
        """Save model registry to disk"""
        data = {}
        for model_id, config in self.models.items():
            data[model_id] = {
                "name": config.name,
                "model_type": config.model_type.value,
                "provider": config.provider.value,
                "description": config.description,
                "version": config.version,
                "url": config.url,
                "hash": config.hash,
                "size": config.size,
                "requirements": config.requirements,
                "capabilities": config.capabilities,
                "parameters": config.parameters,
                "metadata": config.metadata,
                "is_downloaded": config.is_downloaded,
                "last_validated": config.last_validated.isoformat()
                if config.last_validated
                else None,
                "last_benchmarked": config.last_benchmarked.isoformat()
                if config.last_benchmarked
                else None
            }
        
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)

    def register_model(
        self,
        model_id: str,
        config: ModelConfig
    ) -> None:
        """Register a new model or update existing registration"""
        self.models[model_id] = config
        self._save_registry()

    def unregister_model(self, model_id: str) -> bool:
        """Remove a model from the registry"""
        if model_id in self.models:
            del self.models[model_id]
            self._save_registry()
            return True
        return False

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get model configuration by ID"""
        return self.models.get(model_id)

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        provider: Optional[ModelProvider] = None,
        capability: Optional[str] = None,
        downloaded_only: bool = False
    ) -> List[str]:
        """List models with optional filtering"""
        models = []
        for model_id, config in self.models.items():
            if model_type and config.model_type != model_type:
                continue
            if provider and config.provider != provider:
                continue
            if capability and capability not in config.capabilities:
                continue
            if downloaded_only and not config.is_downloaded:
                continue
            models.append(model_id)
        return models

    def update_model_status(
        self,
        model_id: str,
        is_downloaded: Optional[bool] = None,
        last_validated: Optional[datetime] = None,
        last_benchmarked: Optional[datetime] = None
    ) -> bool:
        """Update model status information"""
        if model_id not in self.models:
            return False

        config = self.models[model_id]
        if is_downloaded is not None:
            config.is_downloaded = is_downloaded
        if last_validated is not None:
            config.last_validated = last_validated
        if last_benchmarked is not None:
            config.last_benchmarked = last_benchmarked

        self._save_registry()
        return True

    def get_model_requirements(
        self,
        model_ids: Union[str, List[str]]
    ) -> Dict[str, str]:
        """Get combined requirements for specified models"""
        if isinstance(model_ids, str):
            model_ids = [model_ids]

        requirements = {}
        for model_id in model_ids:
            if model_id not in self.models:
                continue
            
            model_reqs = self.models[model_id].requirements
            for package, version in model_reqs.items():
                if package not in requirements:
                    requirements[package] = version
                else:
                    # Keep the higher version requirement
                    current = requirements[package].replace(">=", "")
                    new = version.replace(">=", "")
                    if current < new:
                        requirements[package] = version

        return requirements

    def find_models_by_capability(
        self,
        capability: str,
        model_type: Optional[ModelType] = None,
        downloaded_only: bool = False
    ) -> List[str]:
        """Find models that have a specific capability"""
        models = []
        for model_id, config in self.models.items():
            if capability in config.capabilities:
                if model_type and config.model_type != model_type:
                    continue
                if downloaded_only and not config.is_downloaded:
                    continue
                models.append(model_id)
        return models

    def get_model_dependencies(
        self,
        model_id: str
    ) -> Dict[str, List[str]]:
        """Get model dependencies and related models"""
        if model_id not in self.models:
            return {}

        config = self.models[model_id]
        dependencies = {
            "requirements": list(config.requirements.keys()),
            "related_models": []
        }

        # Find related models with similar capabilities
        for capability in config.capabilities:
            for other_id, other_config in self.models.items():
                if (
                    other_id != model_id
                    and capability in other_config.capabilities
                    and other_id not in dependencies["related_models"]
                ):
                    dependencies["related_models"].append(other_id)

        return dependencies

    def validate_config(self, config: ModelConfig) -> List[str]:
        """Validate model configuration"""
        errors = []

        # Check required fields
        if not config.name:
            errors.append("Model name is required")
        if not config.version:
            errors.append("Model version is required")
        if not config.url:
            errors.append("Model URL is required")
        if not config.hash:
            errors.append("Model hash is required")
        if config.size <= 0:
            errors.append("Model size must be positive")
        if not config.capabilities:
            errors.append("At least one capability must be specified")

        # Validate URL format
        if not config.url.startswith(("http://", "https://", "file://")):
            errors.append("Invalid URL format")

        # Validate requirements format
        for package, version in config.requirements.items():
            if not version.startswith(">="):
                errors.append(f"Invalid version format for {package}, must use '>='")

        return errors

    def export_registry(self, output_file: Path) -> bool:
        """Export registry to a file"""
        try:
            with open(output_file, "w") as f:
                json.dump(
                    {
                        model_id: {
                            "name": config.name,
                            "type": config.model_type.value,
                            "provider": config.provider.value,
                            "version": config.version,
                            "capabilities": config.capabilities,
                            "is_downloaded": config.is_downloaded
                        }
                        for model_id, config in self.models.items()
                    },
                    f,
                    indent=2
                )
            return True
        except Exception as e:
            logger.error(f"Error exporting registry: {e}")
            return False

    def import_registry(self, input_file: Path) -> bool:
        """Import registry from a file"""
        if not input_file.exists():
            return False

        try:
            with open(input_file, "r") as f:
                data = json.load(f)
                for model_id, config in data.items():
                    if model_id in self.models:
                        logger.warning(f"Skipping existing model: {model_id}")
                        continue

                    self.models[model_id] = ModelConfig(
                        name=config["name"],
                        model_type=ModelType(config["type"]),
                        provider=ModelProvider(config["provider"]),
                        description=config.get("description", ""),
                        version=config["version"],
                        url=config.get("url", ""),
                        hash=config.get("hash", ""),
                        size=config.get("size", 0),
                        requirements=config.get("requirements", {}),
                        capabilities=config["capabilities"],
                        parameters=config.get("parameters", {}),
                        metadata=config.get("metadata", {}),
                        is_downloaded=config.get("is_downloaded", False)
                    )
            
            self._save_registry()
            return True

        except Exception as e:
            logger.error(f"Error importing registry: {e}")
            return False 