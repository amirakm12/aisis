#!/usr/bin/env python3
"""
Advanced Model Download Script with Progress Tracking
Handles downloading AI models with comprehensive progress tracking, error recovery, and memory management
"""

import asyncio
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.model_manager import model_manager, ModelInfo, ModelStatus
from core.memory_manager import memory_manager
from core.error_recovery import error_recovery, ErrorSeverity
from core.config_validator import config_validator, ValidationLevel


# Predefined model configurations
DEFAULT_MODELS = {
    "whisper-base": {
        "model_id": "openai/whisper-base",
        "size_gb": 0.6,
        "description": "Base Whisper model for speech recognition"
    },
    "whisper-small": {
        "model_id": "openai/whisper-small",
        "size_gb": 1.0,
        "description": "Small Whisper model for speech recognition"
    },
    "stable-diffusion-v1-5": {
        "model_id": "runwayml/stable-diffusion-v1-5",
        "size_gb": 4.2,
        "description": "Stable Diffusion v1.5 for image generation"
    },
    "bert-base-uncased": {
        "model_id": "bert-base-uncased",
        "size_gb": 0.4,
        "description": "BERT base model for text processing"
    },
    "gpt2-medium": {
        "model_id": "gpt2-medium",
        "size_gb": 1.5,
        "description": "GPT-2 medium model for text generation"
    },
    "clip-vit-base-patch32": {
        "model_id": "openai/clip-vit-base-patch32",
        "size_gb": 0.6,
        "description": "CLIP model for vision-language tasks"
    },
    "sentence-transformers-all-MiniLM-L6-v2": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "size_gb": 0.1,
        "description": "Sentence transformer for embeddings"
    },
    "facebook-detr-resnet-50": {
        "model_id": "facebook/detr-resnet-50",
        "size_gb": 0.2,
        "description": "DETR object detection model"
    },
    "microsoft-DialoGPT-medium": {
        "model_id": "microsoft/DialoGPT-medium",
        "size_gb": 0.8,
        "description": "DialoGPT for conversational AI"
    },
    "google-flan-t5-base": {
        "model_id": "google/flan-t5-base",
        "size_gb": 0.9,
        "description": "FLAN-T5 base model for instruction following"
    }
}


class ModelDownloader:
    """Advanced model downloader with progress tracking and error handling"""
    
    def __init__(self, models_dir: str = "models", max_concurrent: int = 2):
        """
        Initialize model downloader
        
        Args:
            models_dir: Directory to store models
            max_concurrent: Maximum concurrent downloads
        """
        self.models_dir = Path(models_dir)
        self.max_concurrent = max_concurrent
        self.download_progress = {}
        self.failed_downloads = []
        
        # Initialize managers
        self.model_manager = model_manager
        self.memory_manager = memory_manager
        self.error_recovery = error_recovery
        
        logger.info(f"Model Downloader initialized - Target: {self.models_dir}")
    
    def register_default_models(self):
        """Register all default models"""
        logger.info("Registering default models...")
        
        for name, config in DEFAULT_MODELS.items():
            try:
                self.model_manager.register_model(
                    name=name,
                    model_id=config["model_id"],
                    size_gb=config["size_gb"]
                )
                logger.debug(f"Registered {name}: {config['description']}")
            except Exception as e:
                logger.error(f"Failed to register {name}: {e}")
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their info"""
        models_info = {}
        
        for name, config in DEFAULT_MODELS.items():
            model_info = self.model_manager.get_model_info(name)
            status = model_info.status.value if model_info else "not_registered"
            
            models_info[name] = {
                "description": config["description"],
                "size_gb": config["size_gb"],
                "model_id": config["model_id"],
                "status": status,
                "downloaded": status == "downloaded"
            }
        
        return models_info
    
    def estimate_download_requirements(self, model_names: List[str]) -> Dict[str, Any]:
        """
        Estimate download requirements for selected models
        
        Args:
            model_names: List of model names to download
            
        Returns:
            Requirements analysis
        """
        total_size = 0
        models_to_download = []
        already_downloaded = []
        
        for name in model_names:
            if name not in DEFAULT_MODELS:
                logger.warning(f"Unknown model: {name}")
                continue
            
            model_info = self.model_manager.get_model_info(name)
            if model_info and model_info.status == ModelStatus.DOWNLOADED:
                already_downloaded.append(name)
                continue
            
            config = DEFAULT_MODELS[name]
            total_size += config["size_gb"]
            models_to_download.append({
                "name": name,
                "size_gb": config["size_gb"],
                "model_id": config["model_id"]
            })
        
        # Memory analysis
        memory_stats = self.memory_manager.get_memory_stats()
        memory_analysis = self.memory_manager.can_load_model(total_size)
        
        return {
            "models_to_download": models_to_download,
            "already_downloaded": already_downloaded,
            "total_size_gb": total_size,
            "available_space_gb": memory_stats.available_ram,
            "can_download": memory_analysis["can_load"],
            "memory_recommendations": memory_analysis.get("recommendations", []),
            "estimated_time_minutes": total_size * 2,  # Rough estimate: 2 min per GB
        }
    
    async def download_models(self, 
                            model_names: List[str], 
                            force: bool = False,
                            progress_callback: Optional[callable] = None) -> Dict[str, bool]:
        """
        Download multiple models with progress tracking
        
        Args:
            model_names: List of model names to download
            force: Force re-download if already exists
            progress_callback: Progress callback function
            
        Returns:
            Download results
        """
        logger.info(f"Starting download of {len(model_names)} models")
        
        # Register models if not already registered
        self.register_default_models()
        
        # Estimate requirements
        requirements = self.estimate_download_requirements(model_names)
        logger.info(f"Download requirements: {requirements}")
        
        if not requirements["can_download"]:
            logger.error("Insufficient resources for download")
            for rec in requirements["memory_recommendations"]:
                logger.warning(f"Recommendation: {rec}")
            return {name: False for name in model_names}
        
        # Start memory monitoring
        self.memory_manager.start_monitoring()
        
        # Download models with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        results = {}
        
        async def download_single_model(model_name: str) -> bool:
            async with semaphore:
                try:
                    logger.info(f"Starting download: {model_name}")
                    
                    def progress_wrapper(name, progress, downloaded, total):
                        self.download_progress[name] = {
                            "progress": progress,
                            "downloaded": downloaded,
                            "total": total,
                            "status": "downloading"
                        }
                        if progress_callback:
                            progress_callback(name, progress, downloaded, total)
                    
                    success = await self.model_manager.download_model(
                        model_name=model_name,
                        force=force,
                        progress_callback=progress_wrapper
                    )
                    
                    if success:
                        self.download_progress[model_name]["status"] = "completed"
                        logger.info(f"✅ Successfully downloaded: {model_name}")
                    else:
                        self.download_progress[model_name]["status"] = "failed"
                        self.failed_downloads.append(model_name)
                        logger.error(f"❌ Failed to download: {model_name}")
                    
                    return success
                    
                except Exception as e:
                    logger.error(f"Error downloading {model_name}: {e}")
                    await self.error_recovery.handle_error(
                        e, 
                        {"model_name": model_name, "operation": "download"},
                        ErrorSeverity.HIGH
                    )
                    self.failed_downloads.append(model_name)
                    return False
        
        # Execute downloads
        tasks = [download_single_model(name) for name in model_names if name in DEFAULT_MODELS]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, model_name in enumerate([name for name in model_names if name in DEFAULT_MODELS]):
            if isinstance(results_list[i], bool):
                results[model_name] = results_list[i]
            else:
                results[model_name] = False
                logger.error(f"Exception during {model_name} download: {results_list[i]}")
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        logger.info(f"Download completed: {successful}/{total} successful")
        
        if self.failed_downloads:
            logger.warning(f"Failed downloads: {self.failed_downloads}")
        
        return results
    
    def get_download_status(self) -> Dict[str, Any]:
        """Get current download status"""
        registered_models = self.model_manager.list_models()
        
        status = {
            "registered_models": len(registered_models),
            "downloaded_models": len([m for m in registered_models.values() if m.status == ModelStatus.DOWNLOADED]),
            "failed_downloads": len(self.failed_downloads),
            "active_downloads": len([p for p in self.download_progress.values() if p["status"] == "downloading"]),
            "models": {}
        }
        
        for name, model_info in registered_models.items():
            status["models"][name] = {
                "status": model_info.status.value,
                "size_gb": model_info.size_gb,
                "local_path": model_info.local_path,
                "download_progress": self.download_progress.get(name, {}).get("progress", 0),
                "error_message": model_info.error_message
            }
        
        return status
    
    def cleanup_failed_downloads(self):
        """Clean up failed download artifacts"""
        logger.info("Cleaning up failed downloads...")
        
        for model_name in self.failed_downloads:
            try:
                model_path = self.models_dir / model_name
                if model_path.exists():
                    import shutil
                    shutil.rmtree(model_path)
                    logger.debug(f"Cleaned up {model_path}")
            except Exception as e:
                logger.error(f"Failed to cleanup {model_name}: {e}")
        
        self.failed_downloads.clear()


def print_models_table(models_info: Dict[str, Dict[str, Any]]):
    """Print a formatted table of available models"""
    print("\n📦 Available Models:")
    print("=" * 80)
    print(f"{'Name':<30} {'Size (GB)':<10} {'Status':<12} {'Description'}")
    print("-" * 80)
    
    for name, info in models_info.items():
        status_icon = "✅" if info["downloaded"] else "⬇️"
        print(f"{name:<30} {info['size_gb']:<10.1f} {status_icon} {info['status']:<10} {info['description']}")
    
    print("=" * 80)


def print_download_summary(results: Dict[str, bool], requirements: Dict[str, Any]):
    """Print download summary"""
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    print(f"\n📊 Download Summary:")
    print(f"Successfully downloaded: {successful}/{total} models")
    print(f"Total size: {requirements['total_size_gb']:.1f} GB")
    print(f"Estimated time: {requirements['estimated_time_minutes']:.1f} minutes")
    
    if successful < total:
        failed_models = [name for name, success in results.items() if not success]
        print(f"❌ Failed: {', '.join(failed_models)}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced AI Model Downloader")
    parser.add_argument("--models", "-m", nargs="+", help="Models to download")
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument("--all", "-a", action="store_true", help="Download all models")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download")
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--max-concurrent", type=int, default=2, help="Max concurrent downloads")
    parser.add_argument("--status", "-s", action="store_true", help="Show download status")
    parser.add_argument("--cleanup", action="store_true", help="Clean up failed downloads")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Initialize downloader
    downloader = ModelDownloader(args.models_dir, args.max_concurrent)
    
    # Initialize systems
    memory_manager.start_monitoring()
    error_recovery.start_checkpointing()
    
    try:
        if args.list:
            # List available models
            models_info = downloader.list_available_models()
            print_models_table(models_info)
            
        elif args.status:
            # Show download status
            status = downloader.get_download_status()
            print(f"\n📊 Download Status:")
            print(f"Registered models: {status['registered_models']}")
            print(f"Downloaded models: {status['downloaded_models']}")
            print(f"Failed downloads: {status['failed_downloads']}")
            print(f"Active downloads: {status['active_downloads']}")
            
            if status['models']:
                print("\n📋 Model Details:")
                for name, info in status['models'].items():
                    print(f"  {name}: {info['status']} ({info['size_gb']:.1f}GB)")
            
        elif args.cleanup:
            # Clean up failed downloads
            downloader.cleanup_failed_downloads()
            print("✅ Cleanup completed")
            
        else:
            # Download models
            if args.all:
                model_names = list(DEFAULT_MODELS.keys())
            elif args.models:
                model_names = args.models
            else:
                print("❌ Please specify models to download with --models or use --all")
                return
            
            # Show requirements
            requirements = downloader.estimate_download_requirements(model_names)
            print(f"\n📋 Download Plan:")
            print(f"Models to download: {len(requirements['models_to_download'])}")
            print(f"Already downloaded: {len(requirements['already_downloaded'])}")
            print(f"Total size: {requirements['total_size_gb']:.1f} GB")
            print(f"Available RAM: {requirements['available_space_gb']:.1f} GB")
            print(f"Estimated time: {requirements['estimated_time_minutes']:.1f} minutes")
            
            if not requirements['can_download']:
                print("❌ Insufficient resources!")
                for rec in requirements['memory_recommendations']:
                    print(f"💡 {rec}")
                return
            
            # Confirm download
            if not args.force and requirements['models_to_download']:
                response = input("\n🤔 Proceed with download? (y/N): ")
                if response.lower() != 'y':
                    print("❌ Download cancelled")
                    return
            
            # Progress callback
            def progress_callback(name, progress, downloaded, total):
                print(f"📥 {name}: {progress:.1f}% ({downloaded/1024/1024:.1f}MB/{total/1024/1024:.1f}MB)")
            
            # Download models
            print("\n🚀 Starting downloads...")
            results = await downloader.download_models(
                model_names, 
                force=args.force,
                progress_callback=progress_callback
            )
            
            # Show summary
            print_download_summary(results, requirements)
            
    except KeyboardInterrupt:
        print("\n⚠️ Download interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await error_recovery.handle_error(e, {"operation": "main"}, ErrorSeverity.CRITICAL)
    finally:
        # Cleanup
        memory_manager.stop_monitoring()
        error_recovery.stop_checkpointing()


if __name__ == "__main__":
    asyncio.run(main())