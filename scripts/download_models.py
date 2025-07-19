"""
AISIS Model Download Script
Downloads initial AI models for voice, LLM, and image processing
"""

import os
import sys
import asyncio
from pathlib import Path
from loguru import logger
from huggingface_hub import snapshot_download  # Add this import


def setup_logging():
    """Setup logging for the download script"""
    logger.add("download_models.log", rotation="10 MB", level="INFO")


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
        base_dir / "nerf",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


async def download_whisper_models():
    """Download Whisper ASR models"""
    logger.info("Setting up Whisper models...")

    try:
        import whisper

        # Download different model sizes
        models = ["tiny", "base", "small", "medium"]

        for model_size in models:
            logger.info(f"Downloading Whisper {model_size} model...")
            model = whisper.load_model(model_size)
            logger.info(f"Whisper {model_size} model ready")

    except Exception as e:
        logger.error(f"Failed to setup Whisper models: {e}")


async def download_bark_models():
    """Download Bark TTS models"""
    logger.info("Setting up Bark TTS models...")
    try:
        snapshot_download(repo_id="suno/bark", local_dir="models/bark")
        logger.info("Bark models downloaded")
    except Exception as e:
        logger.error(f"Failed to setup Bark models: {e}")


async def download_llm_models():
    """Download LLM models for orchestrator"""
    logger.info("Setting up LLM models...")
    try:
        snapshot_download(
            repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", local_dir="models/llm/mixtral"
        )
        snapshot_download(repo_id="meta-llama/Meta-Llama-3-8B", local_dir="models/llm/llama3")
        snapshot_download(repo_id="microsoft/Phi-3-mini-4k-instruct", local_dir="models/llm/phi3")
        logger.info("LLM models downloaded")
    except Exception as e:
        logger.error(f"Failed to setup LLM models: {e}")


async def download_diffusion_models():
    """Download diffusion models for generative agent"""
    logger.info("Setting up diffusion models...")
    try:
        snapshot_download(repo_id="stabilityai/sdxl-turbo", local_dir="models/diffusion/sdxl-turbo")
        snapshot_download(repo_id="ai-forever/Kandinsky-3", local_dir="models/diffusion/kandinsky3")
        logger.info("Diffusion models downloaded")
    except Exception as e:
        logger.error(f"Failed to setup diffusion models: {e}")


async def download_restoration_models():
    """Download image restoration models"""
    logger.info("Setting up restoration models...")
    try:
        snapshot_download(repo_id="xinntao/Real-ESRGAN", local_dir="models/restoration/esrgan")
        # Add more as needed, e.g., for inpainting, denoising
        logger.info("Restoration models downloaded")
    except Exception as e:
        logger.error(f"Failed to setup restoration models: {e}")


async def download_style_models():
    """Download style transfer and aesthetic models"""
    logger.info("Setting up style models...")
    try:
        snapshot_download(
            repo_id="nvlabs/stylegan3", local_dir="models/style/stylegan3"
        )  # Note: StyleGAN3 is GitHub, may need git clone instead
        logger.info("Style models downloaded")
    except Exception as e:
        logger.error(f"Failed to setup style models: {e}")


async def download_semantic_models():
    """Download semantic editing models"""
    logger.info("Setting up semantic models...")
    try:
        snapshot_download(repo_id="openai/clip-vit-large-patch14", local_dir="models/semantic/clip")
        logger.info("Semantic models downloaded")
    except Exception as e:
        logger.error(f"Failed to setup semantic models: {e}")


async def download_retouch_models():
    """Download face/body detection and enhancement models"""
    logger.info("Setting up retouch models...")
    try:
        snapshot_download(
            repo_id="yuqinie98/retouch", local_dir="models/retouch"
        )  # Placeholder, find actual repo
        logger.info("Retouch models downloaded")
    except Exception as e:
        logger.error(f"Failed to setup retouch models: {e}")


async def download_nerf_models():
    """Download NeRF models for 3D reconstruction"""
    logger.info("Setting up NeRF models...")
    try:
        snapshot_download(
            repo_id="NVlabs/instant-ngp", local_dir="models/nerf/instant-ngp"
        )  # Note: May need git clone
        logger.info("NeRF models downloaded")
    except Exception as e:
        logger.error(f"Failed to setup NeRF models: {e}")


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
        download_nerf_models(),
    ]

    await asyncio.gather(*tasks, return_exceptions=True)

    logger.info("Model download complete!")
    logger.info("Note: Some models are placeholders and need real implementation")


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
