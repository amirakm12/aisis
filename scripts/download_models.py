#!/usr/bin/env python3
"""
Model download script with progress tracking.
Addresses: Create model download script with progress tracking
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aisis.models.model_loader import model_loader, DownloadProgress
from aisis.core.config import config
from aisis.core.memory_manager import memory_manager
import structlog

logger = structlog.get_logger(__name__)


class ProgressDisplay:
    """Display download progress in terminal."""
    
    def __init__(self):
        self.current_downloads: Dict[str, DownloadProgress] = {}
    
    def update_progress(self, progress: DownloadProgress):
        """Update progress display."""
        self.current_downloads[progress.model_name] = progress
        self._display_progress()
    
    def _display_progress(self):
        """Display current progress."""
        # Clear screen
        print("\033[2J\033[H", end="")
        
        print("🤖 AISIS Model Download Progress")
        print("=" * 50)
        
        if not self.current_downloads:
            print("No active downloads")
            return
        
        for model_name, progress in self.current_downloads.items():
            bar_length = 40
            filled_length = int(bar_length * progress.percentage / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            size_gb = progress.total_size / (1024**3)
            downloaded_gb = progress.downloaded / (1024**3)
            
            print(f"\n📦 {model_name}")
            print(f"[{bar}] {progress.percentage:.1f}%")
            print(f"📊 {downloaded_gb:.2f}GB / {size_gb:.2f}GB")
            
            if progress.speed_mbps > 0:
                print(f"⚡ {progress.speed_mbps:.1f} MB/s", end="")
                if progress.eta_seconds:
                    minutes, seconds = divmod(progress.eta_seconds, 60)
                    print(f" | ETA: {minutes:02d}:{seconds:02d}")
                else:
                    print()
            
            print("-" * 50)
    
    def complete_download(self, model_name: str):
        """Mark download as complete."""
        if model_name in self.current_downloads:
            del self.current_downloads[model_name]
            print(f"✅ {model_name} download completed!")


# Popular models for different use cases
RECOMMENDED_MODELS = {
    "conversational": [
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-large",
        "facebook/blenderbot-400M-distill",
    ],
    "text_generation": [
        "gpt2",
        "gpt2-medium",
        "distilgpt2",
    ],
    "question_answering": [
        "distilbert-base-cased-distilled-squad",
        "bert-large-uncased-whole-word-masking-finetuned-squad",
    ],
    "summarization": [
        "facebook/bart-large-cnn",
        "t5-small",
        "t5-base",
    ],
    "lightweight": [
        "distilbert-base-uncased",
        "distilgpt2",
        "microsoft/DialoGPT-small",
    ]
}


async def download_single_model(model_name: str, progress_display: ProgressDisplay) -> bool:
    """Download a single model with progress tracking."""
    try:
        logger.info(f"Starting download: {model_name}")
        
        # Setup progress callback
        def progress_callback(progress: DownloadProgress):
            progress_display.update_progress(progress)
        
        model_loader.downloader.progress_tracker.add_callback(progress_callback)
        
        try:
            # Download the model
            local_path = await model_loader.download_model(model_name)
            progress_display.complete_download(model_name)
            
            logger.info(f"✅ Successfully downloaded: {model_name}")
            logger.info(f"📁 Location: {local_path}")
            
            return True
            
        finally:
            # Remove callback
            model_loader.downloader.progress_tracker.remove_callback(progress_callback)
            
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {str(e)}")
        return False


async def download_multiple_models(model_names: List[str], max_concurrent: int = 2) -> Dict[str, bool]:
    """Download multiple models with limited concurrency."""
    progress_display = ProgressDisplay()
    
    # Limit concurrent downloads
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def download_with_semaphore(model_name: str) -> tuple[str, bool]:
        async with semaphore:
            success = await download_single_model(model_name, progress_display)
            return model_name, success
    
    # Start all downloads
    tasks = [download_with_semaphore(name) for name in model_names]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    download_results = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Download task failed: {str(result)}")
        else:
            model_name, success = result
            download_results[model_name] = success
    
    return download_results


def list_recommended_models():
    """List recommended models by category."""
    print("🤖 Recommended Models by Category")
    print("=" * 50)
    
    for category, models in RECOMMENDED_MODELS.items():
        print(f"\n📂 {category.upper()}")
        for model in models:
            print(f"  • {model}")
    
    print("\n💡 Usage:")
    print("  python scripts/download_models.py --category conversational")
    print("  python scripts/download_models.py --models gpt2 distilgpt2")
    print("  python scripts/download_models.py --all-recommended")


def check_system_requirements():
    """Check system requirements for model downloads."""
    print("🔍 System Requirements Check")
    print("=" * 30)
    
    # Check memory
    memory_stats = memory_manager.monitor.get_memory_stats()
    print(f"💾 Available RAM: {memory_stats.available_ram_gb:.1f}GB")
    print(f"📊 RAM Usage: {memory_stats.ram_usage_percent:.1f}%")
    
    if memory_stats.available_ram_gb < 4:
        print("⚠️  Warning: Less than 4GB RAM available. Large models may cause issues.")
    
    # Check disk space
    cache_dir = config.model.model_cache_dir
    if cache_dir.exists():
        stat = cache_dir.stat()
        # This is a simplified check - in practice you'd want to check actual disk space
        print(f"📁 Cache Directory: {cache_dir}")
    
    # Check GPU
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(f"🎮 GPU: {gpu.name}")
            print(f"🎮 GPU Memory: {gpu.memoryTotal}MB total, {gpu.memoryFree}MB free")
        else:
            print("🎮 No GPU detected - using CPU only")
    except ImportError:
        print("🎮 GPU info unavailable (GPUtil not installed)")
    
    print()


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download AI models for AISIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download specific models
  python scripts/download_models.py --models gpt2 distilgpt2
  
  # Download all models in a category
  python scripts/download_models.py --category conversational
  
  # Download all recommended models
  python scripts/download_models.py --all-recommended
  
  # List available models
  python scripts/download_models.py --list
  
  # Check system requirements
  python scripts/download_models.py --check-system
        """
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model names to download"
    )
    
    parser.add_argument(
        "--category",
        choices=RECOMMENDED_MODELS.keys(),
        help="Download all models in a category"
    )
    
    parser.add_argument(
        "--all-recommended",
        action="store_true",
        help="Download all recommended models"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recommended models"
    )
    
    parser.add_argument(
        "--check-system",
        action="store_true",
        help="Check system requirements"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent downloads (default: 2)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model exists"
    )
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.list:
        list_recommended_models()
        return
    
    if args.check_system:
        check_system_requirements()
        return
    
    # Determine models to download
    models_to_download = []
    
    if args.models:
        models_to_download.extend(args.models)
    
    if args.category:
        models_to_download.extend(RECOMMENDED_MODELS[args.category])
    
    if args.all_recommended:
        for category_models in RECOMMENDED_MODELS.values():
            models_to_download.extend(category_models)
    
    if not models_to_download:
        print("❌ No models specified. Use --help for usage information.")
        return
    
    # Remove duplicates while preserving order
    models_to_download = list(dict.fromkeys(models_to_download))
    
    print(f"🚀 Starting download of {len(models_to_download)} models...")
    print(f"📂 Cache directory: {config.model.model_cache_dir}")
    print(f"⚡ Max concurrent downloads: {args.max_concurrent}")
    print()
    
    # Check system requirements first
    memory_stats = memory_manager.monitor.get_memory_stats()
    if memory_stats.available_ram_gb < 2:
        print("⚠️  Warning: Low memory detected. Consider closing other applications.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Start memory monitoring
    memory_manager.start()
    
    try:
        # Download models
        results = await download_multiple_models(models_to_download, args.max_concurrent)
        
        # Summary
        successful = [model for model, success in results.items() if success]
        failed = [model for model, success in results.items() if not success]
        
        print("\n" + "=" * 50)
        print("📊 Download Summary")
        print("=" * 50)
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            print("\n✅ Successfully downloaded:")
            for model in successful:
                print(f"  • {model}")
        
        if failed:
            print("\n❌ Failed downloads:")
            for model in failed:
                print(f"  • {model}")
        
        # Cache info
        cache_info = model_loader.get_cache_status()
        print(f"\n📦 Total models in cache: {len(cache_info['cached_models'])}")
        print(f"💾 Cache size: {cache_info['total_size_gb']:.2f}GB")
        
    finally:
        memory_manager.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)