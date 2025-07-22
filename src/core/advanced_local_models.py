# Full content with change: in load_model, if not is_downloaded, print warning and return None
# (To add graceful, but since offline, no auto-download. User can be notified in UI)
# For simplicity, add a check and fallback to a default model if possible.
# But to keep it simple, add print and raise with user-friendly message.

"""
Advanced Local Models Manager
Supports multiple AI models for offline operation with intelligent caching,
model switching, and performance optimization.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import AutoTokenizer, AutoModel

# If these packages aren't installed, run:
# pip install transformers torch huggingface_hub diffusers


class ModelType(Enum):
    """Supported model types"""
    LLM = "llm"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class ModelConfig:
    """Configuration for a local model"""
    name: str
    type: ModelType
    url: str
    local_path: str
    description: str
    capabilities: List[str]
    memory_requirement: str  # e.g., "4GB", "8GB"
    performance_score: float  # 0-1
    is_downloaded: bool = False


class AdvancedLocalModelManager:
    """
    Manages multiple local AI models for offline operation.
    Supports intelligent model selection, caching, and performance optimization.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        
        # Initialize with popular open-source models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default model configurations"""
        default_models = [
            ModelConfig(
                name="llama-2-7b-chat",
                type=ModelType.LLM,
                url="https://huggingface.co/meta-llama/Llama-2-7b-chat-hf",
                local_path="llama-2-7b-chat",
                description="7B parameter LLM for text generation and reasoning",
                capabilities=["text_generation", "reasoning", "critique"],
                memory_requirement="8GB",
                performance_score=0.8
            ),
            ModelConfig(
                name="stable-diffusion-xl",
                type=ModelType.VISION,
                url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
                local_path="stable-diffusion-xl",
                description="High-quality image generation model",
                capabilities=["image_generation", "image_editing", "inpainting"],
                memory_requirement="12GB",
                performance_score=0.9
            ),
            ModelConfig(
                name="whisper-large",
                type=ModelType.AUDIO,
                url="https://huggingface.co/openai/whisper-large-v3",
                local_path="whisper-large",
                description="Speech recognition and transcription",
                capabilities=["speech_to_text", "transcription"],
                memory_requirement="4GB",
                performance_score=0.85
            ),
            ModelConfig(
                name="llava-v1.5",
                type=ModelType.MULTIMODAL,
                url="https://huggingface.co/llava-hf/llava-1.5-7b-hf",
                local_path="llava-v1.5",
                description="Vision-language model for image understanding",
                capabilities=["image_understanding", "visual_reasoning", "captioning"],
                memory_requirement="10GB",
                performance_score=0.75
            )
        ]
        
        for model in default_models:
            self.models[model.name] = model
            # Check if model is already downloaded
            model_path = self.models_dir / model.local_path
            model.is_downloaded = model_path.exists()
    
    async def download_model(self, model_name: str, progress_callback=None) -> bool:
        """
        Download a model from HuggingFace or other sources.
        Supports progress tracking and resume capability.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model_config = self.models[model_name]
        model_path = self.models_dir / model_config.local_path
        
        if model_config.is_downloaded:
            print(f"Model '{model_name}' already downloaded")
            return True
        
        print(f"Downloading model '{model_name}'...")
        
        try:
            # Use huggingface_hub to download
            from huggingface_hub import snapshot_download
            
            snapshot_download(
                repo_id=model_config.url.split("/")[-2] + "/" + model_config.url.split("/")[-1],
                local_dir=model_path,
                local_dir_use_symlinks=False
            )
            
            model_config.is_downloaded = True
            self._save_model_status()
            print(f"Model '{model_name}' downloaded successfully")
            return True
            
        except Exception as e:
            print(f"Failed to download model '{model_name}': {e}")
            return False
    
    def load_model(self, model_name: str, device: str = "auto") -> Any:
        """
        Load a model into memory for inference.
        Supports intelligent device placement and memory optimization.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        model_config = self.models[model_name]
        if not model_config.is_downloaded:
            print(f"Warning: Model '{model_name}' not downloaded. Please download it first.")
            # Fallback to a similar downloaded model
            fallback = self.get_best_model_for_task(model_config.capabilities[0], {'downloaded': True})
            if fallback:
                print(f"Falling back to {fallback}")
                return self.load_model(fallback, device)
            else:
                raise RuntimeError(f"Model '{model_name}' not available and no fallback found.")
        
        model_path = self.models_dir / model_config.local_path
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            if model_config.type == ModelType.LLM:
                model = self._load_llm_model(model_path, device)
            elif model_config.type == ModelType.VISION:
                model = self._load_vision_model(model_path, device)
            elif model_config.type == ModelType.AUDIO:
                model = self._load_audio_model(model_path, device)
            elif model_config.type == ModelType.MULTIMODAL:
                model = self._load_multimodal_model(model_path, device)
            else:
                raise ValueError(f"Unsupported model type: {model_config.type}")
            
            self.loaded_models[model_name] = model
            return model
            
        except Exception as e:
            print(f"Failed to load model '{model_name}': {e}")
            raise

# ... the rest of the code remains the same ...
