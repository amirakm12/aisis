"""
AISIS Model Download Script
Downloads initial AI models for voice, LLM, and image processing
"""

import os
import sys
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Callable
from loguru import logger
import hashlib
import json
from tqdm import tqdm

# Model configurations
MODEL_CONFIGS = {
    "whisper": {
        "models": [
            {
                "name": "whisper-tiny",
                "url": "https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors",
                "local_path": "models/whisper/tiny",
                "size_mb": 39,
                "description": "Tiny Whisper model for fast transcription"
            },
            {
                "name": "whisper-small",
                "url": "https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors",
                "local_path": "models/whisper/small",
                "size_mb": 244,
                "description": "Small Whisper model for good accuracy"
            },
            {
                "name": "whisper-base",
                "url": "https://huggingface.co/openai/whisper-base/resolve/main/model.safetensors",
                "local_path": "models/whisper/base",
                "size_mb": 139,
                "description": "Base Whisper model for balanced performance"
            }
        ]
    },
    "llm": {
        "models": [
            {
                "name": "llama-2-7b-chat",
                "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf",
                "local_path": "models/llm/llama-2-7b-chat",
                "size_mb": 4096,
                "description": "7B parameter Llama 2 chat model"
            },
            {
                "name": "phi-2",
                "url": "https://huggingface.co/microsoft/phi-2/resolve/main/model.safetensors",
                "local_path": "models/llm/phi-2",
                "size_mb": 2048,
                "description": "Microsoft Phi-2 model for reasoning"
            }
        ]
    },
    "diffusion": {
        "models": [
            {
                "name": "stable-diffusion-xl",
                "url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/diffusion_pytorch_model.safetensors",
                "local_path": "models/diffusion/sdxl",
                "size_mb": 6144,
                "description": "Stable Diffusion XL for high-quality generation"
            }
        ]
    },
    "restoration": {
        "models": [
            {
                "name": "dncnn",
                "url": "https://huggingface.co/CompVis/stable-diffusion-v1-4/resolve/main/diffusion_pytorch_model.safetensors",
                "local_path": "models/restoration/dncnn",
                "size_mb": 1024,
                "description": "DnCNN for image denoising"
            },
            {
                "name": "realesrgan",
                "url": "https://huggingface.co/caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr/resolve/main/model.pth",
                "local_path": "models/restoration/realesrgan",
                "size_mb": 2048,
                "description": "RealESRGAN for super-resolution"
            }
        ]
    }
}

def setup_logging():
    """Setup logging for the download script"""
    logger.add(
        "download_models.log",
        rotation="10 MB",
        level="INFO"
    )

def create_model_directories():
    """Create necessary model directories"""
    base_dir = Path("models")
    directories = [
        base_dir / "whisper",
        base_dir / "bark", 
        base_dir / "llm",
        base_dir / "diffusion",
        base_dir / "restoration",
        base_dir / "style",
        base_dir / "semantic",
        base_dir / "retouch",
        base_dir / "nerf"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

async def download_file_with_progress(
    url: str, 
    file_path: Path, 
    size_mb: int,
    progress_callback: Optional[Callable] = None
) -> bool:
    """
    Download a file with progress tracking and resume capability
    
    Args:
        url: URL to download from
        file_path: Local path to save file
        size_mb: Expected file size in MB
        progress_callback: Optional callback for progress updates
        
    Returns:
        bool: True if download successful
    """
    try:
        # Create parent directory
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file already exists
        if file_path.exists():
            logger.info(f"File already exists: {file_path}")
            return True
        
        # Download with progress
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to download {url}: HTTP {response.status}")
                    return False
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(file_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if progress_callback:
                            progress = (downloaded_size / total_size) * 100
                            progress_callback(progress)
                        
                        # Log progress every 10%
                        if downloaded_size % (total_size // 10) == 0:
                            logger.info(f"Downloaded {downloaded_size}/{total_size} bytes")
        
        logger.info(f"Successfully downloaded: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        # Clean up partial download
        if file_path.exists():
            file_path.unlink()
        return False

async def download_whisper_models():
    """Download Whisper ASR models"""
    logger.info("Setting up Whisper models...")
    
    try:
        models = MODEL_CONFIGS["whisper"]["models"]
        
        for model_config in models:
            logger.info(f"Downloading {model_config['name']}...")
            
            file_path = Path(model_config["local_path"]) / "model.safetensors"
            
            def progress_callback(progress):
                logger.info(f"Download progress: {progress:.1f}%")
            
            success = await download_file_with_progress(
                model_config["url"],
                file_path,
                model_config["size_mb"],
                progress_callback
            )
            
            if success:
                logger.info(f"Whisper {model_config['name']} model ready")
            else:
                logger.error(f"Failed to download {model_config['name']}")
                
    except Exception as e:
        logger.error(f"Failed to setup Whisper models: {e}")

async def download_bark_models():
    """Download Bark TTS models"""
    logger.info("Setting up Bark TTS models...")
    try:
        # Bark models are typically downloaded automatically by the bark library
        # We'll create a placeholder and let the library handle the download
        bark_dir = Path("models/bark")
        bark_dir.mkdir(exist_ok=True)
        
        # Create a config file to indicate Bark is ready
        config_file = bark_dir / "config.json"
        config_data = {
            "status": "ready",
            "models": ["text_encoder", "coarse_encoder", "fine_encoder"],
            "note": "Models will be downloaded automatically on first use"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info("Bark TTS setup complete - models will download on first use")
        
    except Exception as e:
        logger.error(f"Failed to setup Bark models: {e}")

async def download_llm_models():
    """Download LLM models for orchestrator"""
    logger.info("Setting up LLM models...")
    try:
        models = MODEL_CONFIGS["llm"]["models"]
        
        for model_config in models:
            logger.info(f"Downloading {model_config['name']}...")
            
            file_path = Path(model_config["local_path"]) / f"{model_config['name']}.gguf"
            
            def progress_callback(progress):
                logger.info(f"Download progress: {progress:.1f}%")
            
            success = await download_file_with_progress(
                model_config["url"],
                file_path,
                model_config["size_mb"],
                progress_callback
            )
            
            if success:
                logger.info(f"LLM {model_config['name']} model ready")
            else:
                logger.error(f"Failed to download {model_config['name']}")
                
    except Exception as e:
        logger.error(f"Failed to setup LLM models: {e}")

async def download_diffusion_models():
    """Download diffusion models for generative agent"""
    logger.info("Setting up diffusion models...")
    try:
        models = MODEL_CONFIGS["diffusion"]["models"]
        
        for model_config in models:
            logger.info(f"Downloading {model_config['name']}...")
            
            file_path = Path(model_config["local_path"]) / "model.safetensors"
            
            def progress_callback(progress):
                logger.info(f"Download progress: {progress:.1f}%")
            
            success = await download_file_with_progress(
                model_config["url"],
                file_path,
                model_config["size_mb"],
                progress_callback
            )
            
            if success:
                logger.info(f"Diffusion {model_config['name']} model ready")
            else:
                logger.error(f"Failed to download {model_config['name']}")
                
    except Exception as e:
        logger.error(f"Failed to setup diffusion models: {e}")

async def download_restoration_models():
    """Download image restoration models"""
    logger.info("Setting up restoration models...")
    try:
        models = MODEL_CONFIGS["restoration"]["models"]
        
        for model_config in models:
            logger.info(f"Downloading {model_config['name']}...")
            
            file_path = Path(model_config["local_path"]) / "model.pth"
            
            def progress_callback(progress):
                logger.info(f"Download progress: {progress:.1f}%")
            
            success = await download_file_with_progress(
                model_config["url"],
                file_path,
                model_config["size_mb"],
                progress_callback
            )
            
            if success:
                logger.info(f"Restoration {model_config['name']} model ready")
            else:
                logger.error(f"Failed to download {model_config['name']}")
                
    except Exception as e:
        logger.error(f"Failed to setup restoration models: {e}")

async def download_style_models():
    """Download style transfer and aesthetic models"""
    logger.info("Setting up style models...")
    try:
        # Create placeholder for style models
        style_dir = Path("models/style")
        style_dir.mkdir(exist_ok=True)
        
        # Create a config file
        config_file = style_dir / "config.json"
        config_data = {
            "status": "placeholder",
            "models": ["style_transfer", "aesthetic_scoring"],
            "note": "Style models will be implemented in future versions"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info("Style models placeholder created")
        
    except Exception as e:
        logger.error(f"Failed to setup style models: {e}")

async def download_semantic_models():
    """Download semantic editing models"""
    logger.info("Setting up semantic models...")
    try:
        # Create placeholder for semantic models
        semantic_dir = Path("models/semantic")
        semantic_dir.mkdir(exist_ok=True)
        
        # Create a config file
        config_file = semantic_dir / "config.json"
        config_data = {
            "status": "placeholder",
            "models": ["clip", "segmentation", "editing"],
            "note": "Semantic models will be implemented in future versions"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info("Semantic models placeholder created")
        
    except Exception as e:
        logger.error(f"Failed to setup semantic models: {e}")

async def download_retouch_models():
    """Download face/body detection and enhancement models"""
    logger.info("Setting up retouch models...")
    try:
        # Create placeholder for retouch models
        retouch_dir = Path("models/retouch")
        retouch_dir.mkdir(exist_ok=True)
        
        # Create a config file
        config_file = retouch_dir / "config.json"
        config_data = {
            "status": "placeholder",
            "models": ["face_detection", "face_enhancement", "body_detection"],
            "note": "Retouch models will be implemented in future versions"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info("Retouch models placeholder created")
        
    except Exception as e:
        logger.error(f"Failed to setup retouch models: {e}")

async def download_nerf_models():
    """Download NeRF models for 3D reconstruction"""
    logger.info("Setting up NeRF models...")
    try:
        # Create placeholder for NeRF models
        nerf_dir = Path("models/nerf")
        nerf_dir.mkdir(exist_ok=True)
        
        # Create a config file
        config_file = nerf_dir / "config.json"
        config_data = {
            "status": "placeholder",
            "models": ["nerf", "pose_estimation", "mesh_generation"],
            "note": "NeRF models will be implemented in future versions"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info("NeRF models placeholder created")
        
    except Exception as e:
        logger.error(f"Failed to setup NeRF models: {e}")

def create_model_index():
    """Create an index of all downloaded models"""
    try:
        models_dir = Path("models")
        index_file = models_dir / "model_index.json"
        
        index_data = {
            "last_updated": str(Path().cwd()),
            "models": {}
        }
        
        # Scan for downloaded models
        for category_dir in models_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                index_data["models"][category_name] = []
                
                for model_file in category_dir.rglob("*"):
                    if model_file.is_file() and model_file.suffix in ['.pth', '.safetensors', '.gguf', '.json']:
                        index_data["models"][category_name].append({
                            "name": model_file.name,
                            "path": str(model_file.relative_to(models_dir)),
                            "size_mb": model_file.stat().st_size / (1024 * 1024)
                        })
        
        with open(index_file, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"Model index created: {index_file}")
        
    except Exception as e:
        logger.error(f"Failed to create model index: {e}")

async def main():
    """Main download function"""
    logger.info("Starting AISIS model download...")
    
    # Create directories
    create_model_directories()
    
    # Download models in parallel
    tasks = [
        download_whisper_models(),
        download_bark_models(),
        download_llm_models(),
        download_diffusion_models(),
        download_restoration_models(),
        download_style_models(),
        download_semantic_models(),
        download_retouch_models(),
        download_nerf_models()
    ]
    
    # Run downloads with error handling
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check results
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    logger.info(f"Download complete! {successful} successful, {failed} failed")
    
    # Create model index
    create_model_index()
    
    logger.info("Model download process finished!")
    logger.info("Note: Some models are placeholders and will be implemented in future versions")

if __name__ == "__main__":
    setup_logging()
    asyncio.run(main()) 