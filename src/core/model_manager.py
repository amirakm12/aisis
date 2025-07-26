"""
Advanced Model Management System
Handles model downloading, loading, and memory-safe operations
"""

import os
import json
import hashlib
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from tqdm import tqdm

try:
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from huggingface_hub import hf_hub_download, snapshot_download
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers/PyTorch not available - model loading disabled")

from .memory_manager import memory_manager


class ModelStatus(Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class ModelInfo:
    """Model information container"""
    name: str
    model_id: str
    size_gb: float
    status: ModelStatus
    local_path: Optional[str] = None
    download_progress: float = 0.0
    error_message: Optional[str] = None
    last_used: Optional[str] = None
    memory_usage: Optional[float] = None
    quantization: Optional[str] = None


@dataclass
class DownloadProgress:
    """Download progress tracking"""
    model_name: str
    total_size: int
    downloaded: int
    speed: float
    eta: float
    status: str


class ModelManager:
    """
    Advanced model management with download tracking and memory safety
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 cache_dir: str = "model_cache",
                 max_concurrent_downloads: int = 2,
                 max_loaded_models: int = 3):
        """
        Initialize model manager
        
        Args:
            models_dir: Directory to store downloaded models
            cache_dir: Cache directory for temporary files
            max_concurrent_downloads: Maximum concurrent downloads
            max_loaded_models: Maximum models to keep in memory
        """
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.max_concurrent_downloads = max_concurrent_downloads
        self.max_loaded_models = max_loaded_models
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.download_progress: Dict[str, DownloadProgress] = {}
        
        # Threading
        self.download_executor = ThreadPoolExecutor(max_workers=max_concurrent_downloads)
        self.load_executor = ThreadPoolExecutor(max_workers=2)
        
        # Callbacks
        self.progress_callbacks: List[Callable] = []
        self.status_callbacks: List[Callable] = []
        
        # Load existing model registry
        self._load_registry()
        
        logger.info(f"Model Manager initialized - Models dir: {self.models_dir}")
    
    def register_model(self, 
                      name: str, 
                      model_id: str, 
                      size_gb: float,
                      quantization: Optional[str] = None) -> ModelInfo:
        """
        Register a model for management
        
        Args:
            name: Model name/identifier
            model_id: HuggingFace model ID or path
            size_gb: Estimated model size in GB
            quantization: Quantization method if applicable
            
        Returns:
            Model info object
        """
        model_info = ModelInfo(
            name=name,
            model_id=model_id,
            size_gb=size_gb,
            status=ModelStatus.NOT_DOWNLOADED,
            quantization=quantization
        )
        
        # Check if already downloaded
        local_path = self.models_dir / name
        if local_path.exists():
            model_info.status = ModelStatus.DOWNLOADED
            model_info.local_path = str(local_path)
        
        self.models[name] = model_info
        self._save_registry()
        
        logger.info(f"Registered model: {name} ({size_gb:.1f}GB)")
        return model_info
    
    async def download_model(self, 
                           model_name: str, 
                           force: bool = False,
                           progress_callback: Optional[Callable] = None) -> bool:
        """
        Download a model with progress tracking
        
        Args:
            model_name: Name of model to download
            force: Force re-download if already exists
            progress_callback: Progress callback function
            
        Returns:
            Success status
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        
        model_info = self.models[model_name]
        
        if model_info.status == ModelStatus.DOWNLOADED and not force:
            logger.info(f"Model {model_name} already downloaded")
            return True
        
        if model_info.status == ModelStatus.DOWNLOADING:
            logger.warning(f"Model {model_name} already downloading")
            return False
        
        # Check memory availability
        memory_check = memory_manager.can_load_model(model_info.size_gb)
        if not memory_check['can_load']:
            logger.error(f"Insufficient memory to download {model_name}: {memory_check}")
            return False
        
        # Update status
        model_info.status = ModelStatus.DOWNLOADING
        model_info.error_message = None
        self._notify_status_change(model_name, model_info.status)
        
        try:
            # Create model directory
            model_path = self.models_dir / model_name
            model_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting download of {model_name}")
            
            if TRANSFORMERS_AVAILABLE:
                # Use HuggingFace hub for downloading
                await self._download_hf_model(model_info, model_path, progress_callback)
            else:
                # Fallback download method
                await self._download_generic_model(model_info, model_path, progress_callback)
            
            # Update model info
            model_info.status = ModelStatus.DOWNLOADED
            model_info.local_path = str(model_path)
            model_info.download_progress = 100.0
            
            self._save_registry()
            self._notify_status_change(model_name, model_info.status)
            
            logger.info(f"Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            model_info.status = ModelStatus.ERROR
            model_info.error_message = str(e)
            self._notify_status_change(model_name, model_info.status)
            return False
    
    async def _download_hf_model(self, 
                                model_info: ModelInfo, 
                                model_path: Path,
                                progress_callback: Optional[Callable] = None):
        """Download model from HuggingFace Hub"""
        
        def progress_hook(downloaded: int, total: int):
            progress = (downloaded / total) * 100 if total > 0 else 0
            model_info.download_progress = progress
            
            if progress_callback:
                progress_callback(model_info.name, progress, downloaded, total)
            
            self._notify_progress_change(model_info.name, progress)
        
        # Download model using snapshot_download for full model
        try:
            snapshot_download(
                repo_id=model_info.model_id,
                local_dir=str(model_path),
                resume_download=True,
                local_files_only=False
            )
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            raise
    
    async def _download_generic_model(self, 
                                    model_info: ModelInfo, 
                                    model_path: Path,
                                    progress_callback: Optional[Callable] = None):
        """Generic model download method"""
        # This would implement custom download logic for non-HF models
        logger.warning("Generic download not implemented - using placeholder")
        
        # Simulate download progress
        for i in range(101):
            model_info.download_progress = i
            if progress_callback:
                progress_callback(model_info.name, i, i, 100)
            await asyncio.sleep(0.1)
    
    async def load_model(self, 
                        model_name: str,
                        device: Optional[str] = None,
                        quantization: Optional[str] = None) -> Any:
        """
        Load a model into memory with safety checks
        
        Args:
            model_name: Name of model to load
            device: Target device (cuda, cpu, auto)
            quantization: Quantization method
            
        Returns:
            Loaded model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")
        
        model_info = self.models[model_name]
        
        if model_info.status != ModelStatus.DOWNLOADED:
            raise ValueError(f"Model {model_name} not downloaded")
        
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        # Check memory before loading
        memory_check = memory_manager.can_load_model(model_info.size_gb)
        if not memory_check['can_load']:
            # Try cleanup first
            logger.warning(f"Memory pressure detected, attempting cleanup before loading {model_name}")
            memory_manager.cleanup_memory()
            
            # Check again
            memory_check = memory_manager.can_load_model(model_info.size_gb)
            if not memory_check['can_load']:
                # Unload least recently used models
                await self._free_memory_for_model(model_info.size_gb)
        
        # Update status
        model_info.status = ModelStatus.LOADING
        self._notify_status_change(model_name, model_info.status)
        
        try:
            logger.info(f"Loading model {model_name}")
            
            if TRANSFORMERS_AVAILABLE:
                model = await self._load_transformers_model(model_info, device, quantization)
            else:
                raise RuntimeError("Transformers not available for model loading")
            
            # Store loaded model
            self.loaded_models[model_name] = model
            model_info.status = ModelStatus.LOADED
            
            # Update memory usage
            if hasattr(model, 'get_memory_footprint'):
                model_info.memory_usage = model.get_memory_footprint() / (1024**3)  # GB
            
            self._save_registry()
            self._notify_status_change(model_name, model_info.status)
            
            logger.info(f"Successfully loaded {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            model_info.status = ModelStatus.ERROR
            model_info.error_message = str(e)
            self._notify_status_change(model_name, model_info.status)
            raise
    
    async def _load_transformers_model(self, 
                                     model_info: ModelInfo, 
                                     device: Optional[str] = None,
                                     quantization: Optional[str] = None):
        """Load model using transformers library"""
        
        model_path = model_info.local_path
        
        # Load configuration
        config = AutoConfig.from_pretrained(model_path)
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model with appropriate settings
        load_kwargs = {
            "pretrained_model_name_or_path": model_path,
            "config": config,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if device == "cuda" else None,
            "low_cpu_mem_usage": True
        }
        
        # Add quantization if specified
        if quantization:
            if quantization == "8bit":
                load_kwargs["load_in_8bit"] = True
            elif quantization == "4bit":
                load_kwargs["load_in_4bit"] = True
        
        # Load model
        model = AutoModel.from_pretrained(**load_kwargs)
        
        return model
    
    async def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory
        
        Args:
            model_name: Name of model to unload
            
        Returns:
            Success status
        """
        if model_name not in self.loaded_models:
            logger.warning(f"Model {model_name} not loaded")
            return False
        
        try:
            # Remove from loaded models
            del self.loaded_models[model_name]
            
            # Update status
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.DOWNLOADED
                self.models[model_name].memory_usage = None
                self._notify_status_change(model_name, ModelStatus.UNLOADED)
            
            # Force garbage collection
            memory_manager.cleanup_memory()
            
            logger.info(f"Unloaded model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload {model_name}: {e}")
            return False
    
    async def _free_memory_for_model(self, required_gb: float):
        """Free memory by unloading least recently used models"""
        logger.info(f"Freeing memory for model requiring {required_gb:.1f}GB")
        
        # Sort models by last used (oldest first)
        loaded_models = [(name, info) for name, info in self.models.items() 
                        if name in self.loaded_models]
        
        # Unload models until we have enough memory
        for model_name, _ in loaded_models:
            await self.unload_model(model_name)
            
            # Check if we have enough memory now
            memory_check = memory_manager.can_load_model(required_gb)
            if memory_check['can_load']:
                break
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get model information"""
        return self.models.get(model_name)
    
    def list_models(self) -> Dict[str, ModelInfo]:
        """List all registered models"""
        return self.models.copy()
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    def add_progress_callback(self, callback: Callable):
        """Add progress callback"""
        self.progress_callbacks.append(callback)
    
    def add_status_callback(self, callback: Callable):
        """Add status change callback"""
        self.status_callbacks.append(callback)
    
    def _notify_progress_change(self, model_name: str, progress: float):
        """Notify progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(model_name, progress)
            except Exception as e:
                logger.error(f"Progress callback failed: {e}")
    
    def _notify_status_change(self, model_name: str, status: ModelStatus):
        """Notify status callbacks"""
        for callback in self.status_callbacks:
            try:
                callback(model_name, status)
            except Exception as e:
                logger.error(f"Status callback failed: {e}")
    
    def _load_registry(self):
        """Load model registry from disk"""
        registry_file = self.models_dir / "registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    data = json.load(f)
                
                for name, model_data in data.items():
                    model_info = ModelInfo(**model_data)
                    # Convert status back to enum
                    model_info.status = ModelStatus(model_info.status)
                    self.models[name] = model_info
                
                logger.info(f"Loaded {len(self.models)} models from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
    
    def _save_registry(self):
        """Save model registry to disk"""
        registry_file = self.models_dir / "registry.json"
        try:
            data = {}
            for name, model_info in self.models.items():
                model_data = asdict(model_info)
                model_data['status'] = model_info.status.value
                data[name] = model_data
            
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up model manager")
        
        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            await self.unload_model(model_name)
        
        # Shutdown executors
        self.download_executor.shutdown(wait=True)
        self.load_executor.shutdown(wait=True)


# Global model manager instance
model_manager = ModelManager()


def setup_model_management(models_dir: str = "models") -> ModelManager:
    """
    Setup global model management
    
    Args:
        models_dir: Directory for storing models
        
    Returns:
        Configured model manager
    """
    global model_manager
    model_manager = ModelManager(models_dir=models_dir)
    return model_manager