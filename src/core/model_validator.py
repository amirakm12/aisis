"""
Model Validation System
Handles model validation, integrity checks, and compatibility testing
"""

import os
import json
import torch
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from enum import Enum


class ValidationLevel(Enum):
    """Validation levels for model checks"""

    BASIC = "basic"  # Hash and metadata checks
    STANDARD = "standard"  # Basic + model loading and basic inference
    THOROUGH = "thorough"  # Standard + comprehensive testing
    STRICT = "strict"  # Thorough + security and vulnerability checks


@dataclass
class ValidationResult:
    """Results of model validation"""

    is_valid: bool
    validation_level: ValidationLevel
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime


class ModelValidator:
    """Handles comprehensive model validation and integrity checks"""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.validation_cache_dir = models_dir / ".validation_cache"
        self.validation_cache_dir.mkdir(exist_ok=True)
        self._load_validation_cache()

    def _load_validation_cache(self) -> None:
        """Load validation results cache"""
        self.validation_cache: Dict[str, Dict[str, ValidationResult]] = {}
        cache_file = self.validation_cache_dir / "validation_results.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    for model_id, versions in data.items():
                        self.validation_cache[model_id] = {}
                        for version, result in versions.items():
                            self.validation_cache[model_id][version] = ValidationResult(
                                is_valid=result["is_valid"],
                                validation_level=ValidationLevel(result["validation_level"]),
                                checks_passed=result["checks_passed"],
                                checks_failed=result["checks_failed"],
                                warnings=result["warnings"],
                                metadata=result["metadata"],
                                timestamp=datetime.fromisoformat(result["timestamp"]),
                            )
            except Exception as e:
                logger.error(f"Error loading validation cache: {e}")
                self.validation_cache = {}

    def _save_validation_cache(self) -> None:
        """Save validation results cache"""
        cache_file = self.validation_cache_dir / "validation_results.json"
        data = {}

        for model_id, versions in self.validation_cache.items():
            data[model_id] = {}
            for version, result in versions.items():
                data[model_id][version] = {
                    "is_valid": result.is_valid,
                    "validation_level": result.validation_level.value,
                    "checks_passed": result.checks_passed,
                    "checks_failed": result.checks_failed,
                    "warnings": result.warnings,
                    "metadata": result.metadata,
                    "timestamp": result.timestamp.isoformat(),
                }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

    async def validate_model(
        self,
        model_id: str,
        version: str,
        expected_hash: str,
        model_type: str,
        level: ValidationLevel = ValidationLevel.STANDARD,
        force: bool = False,
    ) -> ValidationResult:
        """
        Validate a model file with comprehensive checks

        Args:
            model_id: Unique identifier for the model
            version: Version string
            expected_hash: Expected SHA-256 hash
            model_type: Type of model (e.g., "llm", "vision", etc.)
            level: Validation level to perform
            force: Force revalidation even if cached result exists

        Returns:
            ValidationResult containing validation details
        """
        # Check cache first
        cache_key = f"{model_id}-{version}"
        if not force and cache_key in self.validation_cache.get(model_id, {}):
            cached = self.validation_cache[model_id][version]
            if cached.validation_level.value >= level.value:
                return cached

        checks_passed = []
        checks_failed = []
        warnings = []
        metadata = {}

        # Basic validation (always performed)
        model_path = self.models_dir / f"{model_id}-{version}"

        # Check file existence
        if not model_path.exists():
            checks_failed.append("file_exists")
            return ValidationResult(
                is_valid=False,
                validation_level=level,
                checks_passed=checks_passed,
                checks_failed=checks_failed,
                warnings=warnings,
                metadata=metadata,
                timestamp=datetime.now(),
            )
        checks_passed.append("file_exists")

        # Check file size
        file_size = model_path.stat().st_size
        if file_size == 0:
            checks_failed.append("file_size")
        else:
            checks_passed.append("file_size")
            metadata["file_size"] = file_size

        # Verify hash
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        if sha256_hash.hexdigest() != expected_hash:
            checks_failed.append("hash_match")
        else:
            checks_passed.append("hash_match")

        # Standard validation
        if level.value >= ValidationLevel.STANDARD.value and not checks_failed:
            try:
                if model_type == "pytorch":
                    model = torch.load(model_path, map_location="cpu")
                    checks_passed.append("model_load")

                    # Basic model structure checks
                    if hasattr(model, "state_dict"):
                        checks_passed.append("has_state_dict")
                        metadata["num_parameters"] = sum(p.numel() for p in model.parameters())
                    else:
                        checks_failed.append("has_state_dict")

                    # Try basic inference
                    try:
                        model.eval()
                        with torch.no_grad():
                            input_shape = (1, 3, 224, 224)  # Default for vision models
                            dummy_input = torch.randn(input_shape)
                            _ = model(dummy_input)
                            checks_passed.append("basic_inference")
                    except Exception as e:
                        checks_failed.append("basic_inference")
                        warnings.append(f"Inference error: {str(e)}")

            except Exception as e:
                checks_failed.append("model_load")
                warnings.append(f"Model loading error: {str(e)}")

        # Create validation result
        result = ValidationResult(
            is_valid=len(checks_failed) == 0,
            validation_level=level,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            metadata=metadata,
            timestamp=datetime.now(),
        )

        # Update cache
        if model_id not in self.validation_cache:
            self.validation_cache[model_id] = {}
        self.validation_cache[model_id][version] = result
        self._save_validation_cache()

        return result

    def get_validation_status(self, model_id: str, version: str) -> Optional[ValidationResult]:
        """Get the cached validation status for a model"""
        return self.validation_cache.get(model_id, {}).get(version)

    def clear_validation_cache(
        self, model_id: Optional[str] = None, version: Optional[str] = None
    ) -> None:
        """Clear validation cache for specific model or all models"""
        if model_id is None:
            self.validation_cache.clear()
        elif version is None and model_id in self.validation_cache:
            del self.validation_cache[model_id]
        elif model_id in self.validation_cache and version in self.validation_cache[model_id]:
            del self.validation_cache[model_id][version]

        self._save_validation_cache()
