"""
Model Zoo for managing available models (vision, LLM, ASR, etc.).
Supports browsing, downloading, updating, and switching models.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from enum import Enum

try:
    from loguru import logger

    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False
    if TYPE_CHECKING:
        from loguru import logger

from .model_manager import ModelManager
from .model_registry import ModelRegistry, ModelType
from .model_benchmarking import ModelBenchmarker, BenchmarkConfig, BenchmarkType
from .error_recovery import error_recovery


class ModelStatus(Enum):
    """Model status in the zoo"""

    AVAILABLE = "available"  # Model is registered but not downloaded
    DOWNLOADED = "downloaded"  # Model is downloaded but not loaded
    LOADED = "loaded"  # Model is loaded in memory
    ACTIVE = "active"  # Model is currently in use
    ERROR = "error"  # Model has an error


class ModelZoo:
    """
    Model Zoo for managing available models (vision, LLM, ASR, etc.).
    Supports browsing, downloading, updating, and switching models.
    """

    def __init__(self, base_dir: Union[str, Path] = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.registry = ModelRegistry(self.base_dir / "registry")
        self.manager = ModelManager(str(self.base_dir / "storage"))
        self.benchmarker = ModelBenchmarker(self.base_dir / "benchmarks")

        # Active models by task
        self.active_models: Dict[str, str] = {}
        self.loaded_models: Dict[str, Any] = {}

    @error_recovery.with_recovery({"module": "model_zoo"})
    async def download_model(self, model_id: str) -> bool:
        """Download a model from remote or local source"""
        try:
            # Get model info from registry
            model_config = self.registry.get_model(model_id)
            if not model_config:
                raise ValueError(f"Model {model_id} not found in registry")

            # Register with manager if needed
            if model_id not in self.manager.versions:
                self.manager.register_model(
                    model_id=model_id,
                    version=model_config.version,
                    url=model_config.url,
                    hash=model_config.hash,
                    metadata=model_config.metadata,
                )

            # Download model
            success = await self.manager.download_model(model_id, model_config.version)
            if success:
                # Update registry status
                self.registry.update_model_status(model_id, is_downloaded=True)
                return True
            return False

        except Exception as e:
            if HAS_LOGGER:
                logger.error(f"Failed to download model {model_id}: {e}")
            return False

    @error_recovery.with_recovery({"module": "model_zoo"})
    async def load_model(self, model_id: str, device: str = "auto") -> Optional[Any]:
        """Load a model into memory"""
        try:
            # Check if already loaded
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]

            # Get model info
            model_config = self.registry.get_model(model_id)
            if not model_config:
                raise ValueError(f"Model {model_id} not found in registry")

            # Download if needed
            # The line `if not model_config.is_downloaded:` is checking if the model associated with the given `model_id` has been downloaded or not. If the model has not been downloaded (`is_downloaded` is `False`), the code inside the `if` block will be executed to download the model before proceeding with further operations like loading the model into memory.
            # The line `if not model_config.is_downloaded:` is checking if the model associated with the given `model_id` has been downloaded or not. If `model_config.is_downloaded` is `False`, it means the model has not been downloaded yet. In that case, the code will proceed to download the model before attempting to load it into memory.
            # The line `if not model_config.is_downloaded:` is checking if the `is_downloaded` attribute of the `model_config` object is `False`. If the model associated with `model_id` has not been downloaded yet, this condition will evaluate to `True`, indicating that the model needs to be downloaded before proceeding with loading it into memory.
            if not model_config.is_downloaded:
                if not await self.download_model(model_id):
                    raise RuntimeError(f"Failed to download model {model_id}")

            # Load model using appropriate method based on type
            model_path = self.manager.models_dir / f"{model_id}-{model_config.version}"
            model = self._load_model_by_type(model_path, model_config.model_type, device)

            # Store in loaded models
            self.loaded_models[model_id] = model
            return model

        except Exception as e:
            if HAS_LOGGER:
                logger.error(f"Failed to load model {model_id}: {e}")
            return None

    def _load_model_by_type(self, model_path: Path, model_type: ModelType, device: str) -> Any:
        """Load model using appropriate method based on type"""
        try:
            # Import optional dependencies
            try:
                import torch
                import transformers
                import diffusers

                HAS_ML_DEPS = True
            except ImportError:
                HAS_ML_DEPS = False
                if TYPE_CHECKING:
                    import torch
                    from transformers import AutoModel, AutoTokenizer
                    from diffusers import StableDiffusionPipeline

            if not HAS_ML_DEPS:
                raise ImportError(
                    "Required ML dependencies not found. Please install: torch, transformers, diffusers"
                )

            if model_type == ModelType.LLM:
                return transformers.AutoModel.from_pretrained(str(model_path))
            elif model_type == ModelType.VISION:
                return torch.load(model_path)
            elif model_type == ModelType.DIFFUSION:
                return diffusers.StableDiffusionPipeline.from_pretrained(str(model_path))
            else:
                # Default to PyTorch loading
                return torch.load(model_path)

        except ImportError as e:
            if HAS_LOGGER:
                logger.error(f"Required dependencies not found: {e}")
            raise

    @error_recovery.with_recovery({"module": "model_zoo"})
    async def switch_model(self, task: str, model_id: str) -> bool:
        """Switch active model for a given task"""
        try:
            # Load new model
            model = await self.load_model(model_id)
            if not model:
                return False

            # Unload previous model if different
            if task in self.active_models:
                prev_model = self.active_models[task]
                if prev_model != model_id and prev_model in self.loaded_models:
                    del self.loaded_models[prev_model]

            # Update active model
            self.active_models[task] = model_id
            return True

        except Exception as e:
            if HAS_LOGGER:
                logger.error(f"Failed to switch model for task {task}: {e}")
            return False

    @error_recovery.with_recovery({"module": "model_zoo"})
    async def benchmark_model(
        self, model_id: str, config: Optional[BenchmarkConfig] = None
    ) -> bool:
        """Run benchmarks on a model"""
        try:
            # Get model info
            model_config = self.registry.get_model(model_id)
            if not model_config:
                raise ValueError(f"Model {model_id} not found in registry")

            # Load model if needed
            model = await self.load_model(model_id)
            if not model:
                return False

            # Use default config if none provided
            if not config:
                config = BenchmarkConfig(
                    type=BenchmarkType.INFERENCE_SPEED, num_runs=100, warmup_runs=10, device="auto"
                )

            # Run benchmarks
            result = await self.benchmarker.run_benchmark(
                model=model, model_id=model_id, version=model_config.version, config=config
            )

            # Update registry with benchmark timestamp
            self.registry.update_model_status(model_id=model_id, last_benchmarked=result.timestamp)

            return True

        except Exception as e:
            if HAS_LOGGER:
                logger.error(f"Failed to benchmark model {model_id}: {e}")
            return False

    def get_model_status(self, model_id: str) -> ModelStatus:
        """Get current status of a model"""
        try:
            if model_id in self.active_models.values():
                return ModelStatus.ACTIVE
            if model_id in self.loaded_models:
                return ModelStatus.LOADED

            model_config = self.registry.get_model(model_id)
            if not model_config:
                raise ValueError(f"Model {model_id} not found in registry")

            if model_config.is_downloaded:
                return ModelStatus.DOWNLOADED
            return ModelStatus.AVAILABLE

        except Exception:
            return ModelStatus.ERROR

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        task: Optional[str] = None,
        status: Optional[ModelStatus] = None,
    ) -> List[Dict[str, Any]]:
        """List models with optional filtering"""
        models = []

        # Get models from registry
        model_ids = self.registry.list_models(model_type=model_type)

        for model_id in model_ids:
            model_config = self.registry.get_model(model_id)
            if not model_config:
                continue

            # Filter by task
            if task and task not in model_config.capabilities:
                continue

            # Get current status
            current_status = self.get_model_status(model_id)

            # Filter by status
            if status and current_status != status:
                continue

            # Get benchmark results if available
            latest_benchmark = self.benchmarker.get_latest_benchmark(model_id, model_config.version)

            models.append(
                {
                    "id": model_id,
                    "name": model_config.name,
                    "type": model_config.model_type.value,
                    "version": model_config.version,
                    "capabilities": model_config.capabilities,
                    "status": current_status.value,
                    "last_benchmarked": model_config.last_benchmarked,
                    "performance_metrics": latest_benchmark.metrics if latest_benchmark else None,
                }
            )

        return models

    def cleanup(self) -> None:
        """Clean up loaded models and free memory"""
        self.loaded_models.clear()
        self.active_models.clear()

        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass


# Global instance
model_zoo = ModelZoo()
