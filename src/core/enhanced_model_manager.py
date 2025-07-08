"""
Enhanced Model Manager
Provides seamless model downloading, loading, and management with agent integration
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from loguru import logger

try:
    import torch
    import requests
    from huggingface_hub import snapshot_download, hf_hub_download
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("HuggingFace transformers not available - using fallback mode")

class ModelStatus(Enum):
    """Model download and loading status"""
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"

@dataclass
class ModelInfo:
    """Enhanced model information"""
    name: str
    repo_id: str
    model_type: str
    description: str
    size_gb: float
    capabilities: List[str]
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    download_progress: float = 0.0
    local_path: Optional[str] = None
    error_message: Optional[str] = None
    checksum: Optional[str] = None

class EnhancedModelManager:
    """
    Enhanced model manager with automatic downloading and agent integration
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.download_threads: Dict[str, threading.Thread] = {}
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        
        self._initialize_model_catalog()
        self._load_status()
    
    def _initialize_model_catalog(self):
        """Initialize the catalog of available models"""
        model_catalog = [
            ModelInfo(
                name="llama-2-7b-chat",
                repo_id="meta-llama/Llama-2-7b-chat-hf",
                model_type="text_generation",
                description="7B parameter conversational AI model",
                size_gb=13.5,
                capabilities=["text_generation", "conversation", "reasoning"]
            ),
            ModelInfo(
                name="stable-diffusion-xl",
                repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                model_type="image_generation",
                description="High-quality text-to-image generation",
                size_gb=6.9,
                capabilities=["image_generation", "text_to_image"]
            ),
            ModelInfo(
                name="whisper-large-v3",
                repo_id="openai/whisper-large-v3",
                model_type="speech_recognition",
                description="State-of-the-art speech recognition",
                size_gb=3.0,
                capabilities=["speech_to_text", "transcription", "translation"]
            ),
            ModelInfo(
                name="llava-1.5-7b",
                repo_id="llava-hf/llava-1.5-7b-hf",
                model_type="vision_language",
                description="Vision-language understanding model",
                size_gb=13.0,
                capabilities=["image_understanding", "visual_reasoning", "captioning"]
            ),
            ModelInfo(
                name="blip-image-captioning",
                repo_id="Salesforce/blip-image-captioning-base",
                model_type="vision_language",
                description="Image captioning and understanding",
                size_gb=1.9,
                capabilities=["image_captioning", "visual_qa"]
            ),
            ModelInfo(
                name="stable-diffusion-v1-5",
                repo_id="runwayml/stable-diffusion-v1-5",
                model_type="image_generation",
                description="Fast and efficient image generation",
                size_gb=4.2,
                capabilities=["image_generation", "text_to_image", "inpainting"]
            ),
            ModelInfo(
                name="gpt2-medium",
                repo_id="gpt2-medium",
                model_type="text_generation",
                description="Medium-sized GPT-2 for text generation",
                size_gb=1.4,
                capabilities=["text_generation", "completion"]
            )
        ]
        
        for model in model_catalog:
            self.models[model.name] = model
            model.local_path = str(self.models_dir / model.name)
    
    def _load_status(self):
        """Load model status from disk"""
        status_file = self.models_dir / "model_status.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                
                for model_name, status_info in status_data.items():
                    if model_name in self.models:
                        model = self.models[model_name]
                        model.status = ModelStatus(status_info.get('status', 'not_downloaded'))
                        model.download_progress = status_info.get('download_progress', 0.0)
                        model.checksum = status_info.get('checksum')
                        
                        # Verify downloaded models still exist
                        if model.status == ModelStatus.DOWNLOADED and model.local_path:
                            if not Path(model.local_path).exists():
                                model.status = ModelStatus.NOT_DOWNLOADED
                                model.download_progress = 0.0
                                
            except Exception as e:
                logger.error(f"Failed to load model status: {e}")
    
    def _save_status(self):
        """Save model status to disk"""
        status_file = self.models_dir / "model_status.json"
        try:
            status_data = {}
            for name, model in self.models.items():
                status_data[name] = {
                    'status': model.status.value,
                    'download_progress': model.download_progress,
                    'checksum': model.checksum
                }
            
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save model status: {e}")
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model"""
        return self.models.get(model_name)
    
    def list_models(self, filter_by_capability: Optional[str] = None) -> List[ModelInfo]:
        """List all models, optionally filtered by capability"""
        models = list(self.models.values())
        
        if filter_by_capability:
            models = [m for m in models if filter_by_capability in m.capabilities]
        
        return models
    
    def get_best_model_for_task(self, task: str, max_size_gb: Optional[float] = None) -> Optional[str]:
        """Find the best model for a specific task"""
        suitable_models = []
        
        for name, model in self.models.items():
            if task in model.capabilities:
                if max_size_gb is None or model.size_gb <= max_size_gb:
                    # Prefer downloaded models
                    priority = 2 if model.status == ModelStatus.DOWNLOADED else 1
                    suitable_models.append((name, model.size_gb, priority))
        
        if not suitable_models:
            return None
        
        # Sort by priority (downloaded first), then by size (smaller first)
        suitable_models.sort(key=lambda x: (-x[2], x[1]))
        return suitable_models[0][0]
    
    def add_progress_callback(self, model_name: str, callback: Callable[[float], None]):
        """Add a progress callback for model downloads"""
        if model_name not in self.progress_callbacks:
            self.progress_callbacks[model_name] = []
        self.progress_callbacks[model_name].append(callback)
    
    def _notify_progress(self, model_name: str, progress: float):
        """Notify progress callbacks"""
        if model_name in self.progress_callbacks:
            for callback in self.progress_callbacks[model_name]:
                try:
                    callback(progress)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
        
        # Update model progress
        if model_name in self.models:
            self.models[model_name].download_progress = progress
    
    async def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download a model asynchronously"""
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model = self.models[model_name]
        
        if not force and model.status == ModelStatus.DOWNLOADED:
            logger.info(f"Model {model_name} already downloaded")
            return True
        
        if model.status == ModelStatus.DOWNLOADING:
            logger.info(f"Model {model_name} is already downloading")
            return False
        
        if not HF_AVAILABLE:
            logger.error("HuggingFace transformers not available - cannot download models")
            model.status = ModelStatus.ERROR
            model.error_message = "HuggingFace transformers not available"
            return False
        
        # Start download in background thread
        model.status = ModelStatus.DOWNLOADING
        model.download_progress = 0.0
        self._save_status()
        
        def download_worker():
            try:
                logger.info(f"Starting download of {model_name}")
                
                                 # Create progress callback
                 def progress_callback(chunk_downloaded, total_size):
                     if total_size > 0:
                         progress = chunk_downloaded / total_size
                         self._notify_progress(model_name, progress)
                 
                 # Download using HuggingFace hub
                 if not model.local_path:
                     raise ValueError(f"No local path set for model {model_name}")
                 local_dir = Path(model.local_path)
                 local_dir.mkdir(parents=True, exist_ok=True)
                
                snapshot_download(
                    repo_id=model.repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                
                # Calculate checksum for verification
                model.checksum = self._calculate_directory_checksum(local_dir)
                model.status = ModelStatus.DOWNLOADED
                model.download_progress = 1.0
                model.error_message = None
                
                logger.info(f"Successfully downloaded {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                model.status = ModelStatus.ERROR
                model.error_message = str(e)
                model.download_progress = 0.0
            
            finally:
                self._save_status()
                if model_name in self.download_threads:
                    del self.download_threads[model_name]
        
        thread = threading.Thread(target=download_worker, daemon=True)
        self.download_threads[model_name] = thread
        thread.start()
        
        return True
    
    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate checksum for a directory"""
        try:
            hasher = hashlib.md5()
            for file_path in sorted(directory.rglob('*')):
                if file_path.is_file():
                    with open(file_path, 'rb') as f:
                        hasher.update(f.read())
            return hasher.hexdigest()
        except Exception:
            return "unknown"
    
    async def load_model(self, model_name: str, device: str = "auto") -> Optional[Any]:
        """Load a model into memory"""
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        model = self.models[model_name]
        
        # Check if already loaded
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Ensure model is downloaded
        if model.status != ModelStatus.DOWNLOADED:
            logger.error(f"Model {model_name} not downloaded")
            return None
        
        if not HF_AVAILABLE:
            logger.error("HuggingFace transformers not available - cannot load models")
            return None
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            model.status = ModelStatus.LOADING
            self._save_status()
            
            logger.info(f"Loading model {model_name} on {device}")
            
            if model.model_type == "text_generation":
                loaded = self._load_text_model(model, device)
            elif model.model_type == "image_generation":
                loaded = self._load_image_generation_model(model, device)
            elif model.model_type == "speech_recognition":
                loaded = self._load_speech_model(model, device)
            elif model.model_type == "vision_language":
                loaded = self._load_vision_language_model(model, device)
            else:
                logger.error(f"Unsupported model type: {model.model_type}")
                model.status = ModelStatus.ERROR
                return None
            
            self.loaded_models[model_name] = loaded
            model.status = ModelStatus.LOADED
            self._save_status()
            
            logger.info(f"Successfully loaded {model_name}")
            return loaded
            
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            model.status = ModelStatus.ERROR
            model.error_message = str(e)
            self._save_status()
            return None
    
    def _load_text_model(self, model: ModelInfo, device: str) -> Dict[str, Any]:
        """Load a text generation model"""
        tokenizer = AutoTokenizer.from_pretrained(model.local_path)
        model_obj = AutoModelForCausalLM.from_pretrained(
            model.local_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None
        )
        
        if device != "cuda":
            model_obj = model_obj.to(device)
        
        return {
            "tokenizer": tokenizer,
            "model": model_obj,
            "type": "text_generation"
        }
    
    def _load_image_generation_model(self, model: ModelInfo, device: str) -> Any:
        """Load an image generation model"""
        from diffusers import StableDiffusionPipeline
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model.local_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        return pipe
    
    def _load_speech_model(self, model: ModelInfo, device: str) -> Dict[str, Any]:
        """Load a speech recognition model"""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        processor = WhisperProcessor.from_pretrained(model.local_path)
        model_obj = WhisperForConditionalGeneration.from_pretrained(model.local_path)
        model_obj = model_obj.to(device)
        
        return {
            "processor": processor,
            "model": model_obj,
            "type": "speech_recognition"
        }
    
    def _load_vision_language_model(self, model: ModelInfo, device: str) -> Dict[str, Any]:
        """Load a vision-language model"""
        if "blip" in model.repo_id.lower():
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            processor = BlipProcessor.from_pretrained(model.local_path)
            model_obj = BlipForConditionalGeneration.from_pretrained(model.local_path)
            model_obj = model_obj.to(device)
            
            return {
                "processor": processor,
                "model": model_obj,
                "type": "blip"
            }
        
        elif "llava" in model.repo_id.lower():
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            
            processor = LlavaProcessor.from_pretrained(model.local_path)
            model_obj = LlavaForConditionalGeneration.from_pretrained(
                model.local_path,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map=device if device == "cuda" else None
            )
            
            return {
                "processor": processor,
                "model": model_obj,
                "type": "llava"
            }
        
        else:
            raise ValueError(f"Unsupported vision-language model: {model.repo_id}")
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            
            if model_name in self.models:
                self.models[model_name].status = ModelStatus.DOWNLOADED
                self._save_status()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Unloaded model {model_name}")
            return True
        
        return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        usage = {"cpu_ram_gb": 0.0, "gpu_vram_gb": 0.0}
        
        if torch.cuda.is_available():
            usage["gpu_vram_gb"] = torch.cuda.memory_allocated() / (1024**3)
        
        return usage
    
    async def cleanup(self):
        """Cleanup all resources"""
        # Stop any running downloads
        for thread in self.download_threads.values():
            if thread.is_alive():
                # Note: Can't cleanly stop threads, but they're daemon threads
                pass
        
        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            self.unload_model(model_name)
        
        self._save_status()
        logger.info("Model manager cleanup complete")

# Global instance
enhanced_model_manager = EnhancedModelManager()