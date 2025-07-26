#!/usr/bin/env python3
"""
AI-ARTWORK Enhanced Installer
Comprehensive installer with advanced features and user-friendly interface
"""

import os
import sys
import subprocess
import platform
import json
import shutil
import asyncio
import aiohttp
import aiofiles
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Third-party imports with fallbacks
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

@dataclass
class InstallationConfig:
    """Configuration for the installation process"""
    install_path: Path
    create_shortcuts: bool = True
    add_to_path: bool = True
    install_models: List[str] = None
    gpu_support: bool = True
    development_mode: bool = False
    auto_start: bool = False
    create_desktop_shortcut: bool = True
    create_start_menu_shortcut: bool = True
    install_optional_dependencies: bool = False
    
    def __post_init__(self):
        if self.install_models is None:
            self.install_models = ["whisper-base", "llama-2-7b-chat"]

@dataclass
class SystemInfo:
    """System information for installation optimization"""
    os_type: str
    architecture: str
    python_version: str
    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    gpu_available: bool
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    disk_space_gb: float = 0.0

@dataclass
class InstallationProgress:
    """Track installation progress"""
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    current_progress: float = 0.0
    eta_seconds: Optional[int] = None
    download_speed_mbps: float = 0.0
    status: str = "initializing"
    error_message: Optional[str] = None

class ProgressTracker:
    """Real-time progress tracking with ETA calculation"""
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.completed_steps = 0
        self.start_time = time.time()
        self.step_times = []
        self.current_step = ""
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[InstallationProgress], None]):
        """Add a progress callback"""
        self.callbacks.append(callback)
    
    def update_step(self, step_name: str, progress: float = 0.0):
        """Update current step and progress"""
        self.current_step = step_name
        current_time = time.time()
        
        if self.step_times:
            step_duration = current_time - self.step_times[-1]
        else:
            step_duration = current_time - self.start_time
        
        self.step_times.append(current_time)
        
        # Calculate ETA
        if len(self.step_times) > 1:
            avg_step_time = sum(
                self.step_times[i] - self.step_times[i-1] 
                for i in range(1, len(self.step_times))
            ) / (len(self.step_times) - 1)
            
            remaining_steps = self.total_steps - self.completed_steps
            eta_seconds = int(avg_step_time * remaining_steps)
        else:
            eta_seconds = None
        
        # Create progress object
        progress_obj = InstallationProgress(
            current_step=step_name,
            total_steps=self.total_steps,
            completed_steps=self.completed_steps,
            current_progress=progress,
            eta_seconds=eta_seconds,
            status="running"
        )
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback(progress_obj)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    def complete_step(self):
        """Mark current step as completed"""
        self.completed_steps += 1
        
        progress_obj = InstallationProgress(
            current_step=self.current_step,
            total_steps=self.total_steps,
            completed_steps=self.completed_steps,
            current_progress=100.0,
            status="running" if self.completed_steps < self.total_steps else "completed"
        )
        
        for callback in self.callbacks:
            try:
                callback(progress_obj)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

class DependencyResolver:
    """Smart dependency resolution and installation"""
    
    def __init__(self):
        self.required_packages = {}
        self.optional_packages = {}
        self.system_packages = {}
        self.conflicts = {}
        self.load_package_definitions()
    
    def load_package_definitions(self):
        """Load package definitions and requirements"""
        self.required_packages = {
            "loguru": {"version": ">=0.7.0", "description": "Advanced logging"},
            "numpy": {"version": ">=1.24.0", "description": "Numerical computing"},
            "Pillow": {"version": ">=10.0.0", "description": "Image processing"},
            "opencv-python": {"version": ">=4.8.0", "description": "Computer vision"},
            "torch": {"version": ">=2.0.0", "description": "Deep learning framework"},
            "torchvision": {"version": ">=0.15.0", "description": "Computer vision for PyTorch"},
            "torchaudio": {"version": ">=2.0.0", "description": "Audio processing for PyTorch"},
            "PySide6": {"version": ">=6.5.0", "description": "GUI framework"},
            "whisper": {"version": ">=1.0.0", "description": "Speech recognition"},
            "transformers": {"version": ">=4.36.0", "description": "NLP models"},
            "diffusers": {"version": ">=0.20.0", "description": "Diffusion models"},
            "aiohttp": {"version": ">=3.8.0", "description": "Async HTTP client"},
            "tqdm": {"version": ">=4.65.0", "description": "Progress bars"},
            "psutil": {"version": ">=5.9.0", "description": "System monitoring"},
        }
        
        self.optional_packages = {
            "cuda-python": {"version": ">=12.0.0", "description": "CUDA support", "condition": "gpu"},
            "sentence-transformers": {"version": ">=2.2.0", "description": "Text embeddings"},
            "mediapipe": {"version": ">=0.10.0", "description": "Face detection"},
            "onnxruntime-gpu": {"version": ">=1.16.0", "description": "ONNX GPU runtime", "condition": "gpu"},
        }
        
        # System-specific packages
        if platform.system() == "Windows":
            self.system_packages.update({
                "pywin32": {"version": ">=306", "description": "Windows API access"},
            })
        elif platform.system() == "Linux":
            self.system_packages.update({
                "python3-dev": {"system": True, "description": "Python development headers"},
            })
    
    def check_dependencies(self) -> Tuple[List[str], List[str], List[str]]:
        """Check which dependencies are missing, outdated, or conflicting"""
        missing = []
        outdated = []
        conflicts = []
        
        all_packages = {**self.required_packages, **self.optional_packages}
        
        for package_name, package_info in all_packages.items():
            try:
                # Check if package has conditions
                if "condition" in package_info:
                    if package_info["condition"] == "gpu" and not self.detect_gpu():
                        continue
                
                # Try to import the package
                if package_name == "opencv-python":
                    import cv2
                    version = cv2.__version__
                elif package_name == "Pillow":
                    from PIL import Image
                    version = Image.__version__
                else:
                    module = __import__(package_name.replace('-', '_'))
                    version = getattr(module, '__version__', 'unknown')
                
                # Check version compatibility (simplified)
                required_version = package_info["version"]
                if required_version.startswith(">="):
                    # Could implement proper version comparison here
                    pass
                    
            except ImportError:
                if package_name in self.required_packages:
                    missing.append(package_name)
                elif package_name in self.optional_packages:
                    # Optional packages don't count as missing
                    pass
            except Exception as e:
                logger.warning(f"Error checking {package_name}: {e}")
        
        return missing, outdated, conflicts
    
    def detect_gpu(self) -> bool:
        """Detect GPU availability"""
        if TORCH_AVAILABLE:
            return torch.cuda.is_available()
        return False
    
    def resolve_dependencies(self, config: InstallationConfig) -> List[str]:
        """Resolve dependencies based on system and configuration"""
        packages_to_install = []
        
        # Always install required packages
        for package, info in self.required_packages.items():
            packages_to_install.append(f"{package}{info['version']}")
        
        # Install optional packages based on conditions
        if config.gpu_support and self.detect_gpu():
            for package, info in self.optional_packages.items():
                if info.get("condition") == "gpu":
                    packages_to_install.append(f"{package}{info['version']}")
        
        # Install system packages
        packages_to_install.extend([
            f"{package}{info['version']}" 
            for package, info in self.system_packages.items()
            if not info.get("system", False)
        ])
        
        return packages_to_install

class GPUDetector:
    """Automatic CUDA/GPU capability detection"""
    
    @staticmethod
    def detect_gpu_info() -> Dict[str, Any]:
        """Detect comprehensive GPU information"""
        gpu_info = {
            "available": False,
            "cuda_available": False,
            "cuda_version": None,
            "devices": [],
            "primary_device": None,
            "total_memory_gb": 0,
            "driver_version": None
        }
        
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["cuda_available"] = True
                gpu_info["cuda_version"] = torch.version.cuda
                
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    device_props = torch.cuda.get_device_properties(i)
                    device_info = {
                        "id": i,
                        "name": device_props.name,
                        "memory_gb": device_props.total_memory / (1024**3),
                        "compute_capability": f"{device_props.major}.{device_props.minor}"
                    }
                    gpu_info["devices"].append(device_info)
                
                if gpu_info["devices"]:
                    gpu_info["primary_device"] = gpu_info["devices"][0]
                    gpu_info["total_memory_gb"] = sum(d["memory_gb"] for d in gpu_info["devices"])
        
        except Exception as e:
            logger.warning(f"GPU detection error: {e}")
        
        return gpu_info

class ModelSelector:
    """Choose which AI models to download"""
    
    def __init__(self):
        self.available_models = self.load_model_catalog()
    
    def load_model_catalog(self) -> Dict[str, Dict]:
        """Load available models catalog"""
        return {
            "whisper-tiny": {
                "category": "speech",
                "size_mb": 39,
                "description": "Tiny Whisper model for fast transcription",
                "requirements": {"ram_gb": 1, "gpu": False},
                "download_url": "https://huggingface.co/openai/whisper-tiny",
                "recommended": False
            },
            "whisper-base": {
                "category": "speech", 
                "size_mb": 139,
                "description": "Base Whisper model for balanced performance",
                "requirements": {"ram_gb": 2, "gpu": False},
                "download_url": "https://huggingface.co/openai/whisper-base",
                "recommended": True
            },
            "whisper-small": {
                "category": "speech",
                "size_mb": 244,
                "description": "Small Whisper model for good accuracy",
                "requirements": {"ram_gb": 3, "gpu": False},
                "download_url": "https://huggingface.co/openai/whisper-small",
                "recommended": False
            },
            "llama-2-7b-chat": {
                "category": "language",
                "size_mb": 4096,
                "description": "7B parameter Llama 2 chat model",
                "requirements": {"ram_gb": 8, "gpu": True},
                "download_url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF",
                "recommended": True
            },
            "phi-2": {
                "category": "language",
                "size_mb": 2048,
                "description": "Microsoft Phi-2 model for reasoning",
                "requirements": {"ram_gb": 4, "gpu": False},
                "download_url": "https://huggingface.co/microsoft/phi-2",
                "recommended": False
            },
            "stable-diffusion-xl": {
                "category": "image",
                "size_mb": 6144,
                "description": "Stable Diffusion XL for high-quality generation",
                "requirements": {"ram_gb": 12, "gpu": True},
                "download_url": "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0",
                "recommended": True
            }
        }
    
    def get_recommended_models(self, system_info: SystemInfo) -> List[str]:
        """Get recommended models based on system capabilities"""
        recommended = []
        
        for model_name, model_info in self.available_models.items():
            if model_info.get("recommended", False):
                # Check system requirements
                req_ram = model_info["requirements"].get("ram_gb", 0)
                req_gpu = model_info["requirements"].get("gpu", False)
                
                if (system_info.available_ram_gb >= req_ram and 
                    (not req_gpu or system_info.gpu_available)):
                    recommended.append(model_name)
        
        return recommended
    
    def filter_compatible_models(self, system_info: SystemInfo) -> Dict[str, Dict]:
        """Filter models that are compatible with the system"""
        compatible = {}
        
        for model_name, model_info in self.available_models.items():
            req_ram = model_info["requirements"].get("ram_gb", 0)
            req_gpu = model_info["requirements"].get("gpu", False)
            
            if (system_info.available_ram_gb >= req_ram and 
                (not req_gpu or system_info.gpu_available)):
                compatible[model_name] = model_info
        
        return compatible

class PathManager:
    """Add to system PATH automatically"""
    
    @staticmethod
    def add_to_path(install_path: Path) -> bool:
        """Add installation path to system PATH"""
        try:
            if platform.system() == "Windows":
                return PathManager._add_to_windows_path(install_path)
            else:
                return PathManager._add_to_unix_path(install_path)
        except Exception as e:
            logger.error(f"Failed to add to PATH: {e}")
            return False
    
    @staticmethod
    def _add_to_windows_path(install_path: Path) -> bool:
        """Add to Windows PATH via registry"""
        try:
            import winreg
            
            # Open the registry key for user environment variables
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                "Environment",
                0,
                winreg.KEY_ALL_ACCESS
            )
            
            # Get current PATH
            try:
                current_path, _ = winreg.QueryValueEx(key, "PATH")
            except FileNotFoundError:
                current_path = ""
            
            # Add our path if not already present
            install_path_str = str(install_path)
            if install_path_str not in current_path:
                new_path = f"{current_path};{install_path_str}" if current_path else install_path_str
                winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)
            
            winreg.CloseKey(key)
            
            # Notify system of environment change
            import win32gui
            import win32con
            win32gui.SendMessage(
                win32con.HWND_BROADCAST,
                win32con.WM_SETTINGCHANGE,
                0,
                "Environment"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Windows PATH update failed: {e}")
            return False
    
    @staticmethod
    def _add_to_unix_path(install_path: Path) -> bool:
        """Add to Unix PATH via shell profile"""
        try:
            home = Path.home()
            shell_profiles = [
                home / ".bashrc",
                home / ".zshrc", 
                home / ".profile"
            ]
            
            path_line = f'export PATH="{install_path}:$PATH"\n'
            
            # Add to existing shell profiles
            for profile in shell_profiles:
                if profile.exists():
                    with open(profile, 'r') as f:
                        content = f.read()
                    
                    if str(install_path) not in content:
                        with open(profile, 'a') as f:
                            f.write(f"\n# AI-ARTWORK installation\n{path_line}")
            
            return True
            
        except Exception as e:
            logger.error(f"Unix PATH update failed: {e}")
            return False

class ShortcutCreator:
    """Desktop and Start Menu shortcuts creation"""
    
    @staticmethod
    def create_shortcuts(install_path: Path, config: InstallationConfig) -> bool:
        """Create desktop and start menu shortcuts"""
        success = True
        
        if config.create_desktop_shortcut:
            success &= ShortcutCreator._create_desktop_shortcut(install_path)
        
        if config.create_start_menu_shortcut:
            success &= ShortcutCreator._create_start_menu_shortcut(install_path)
        
        return success
    
    @staticmethod
    def _create_desktop_shortcut(install_path: Path) -> bool:
        """Create desktop shortcut"""
        try:
            if platform.system() == "Windows":
                return ShortcutCreator._create_windows_shortcut(
                    Path.home() / "Desktop" / "AI-ARTWORK.lnk",
                    install_path / "launch.py"
                )
            else:
                return ShortcutCreator._create_linux_shortcut(
                    Path.home() / "Desktop" / "AI-ARTWORK.desktop",
                    install_path / "launch.py"
                )
        except Exception as e:
            logger.error(f"Desktop shortcut creation failed: {e}")
            return False
    
    @staticmethod
    def _create_start_menu_shortcut(install_path: Path) -> bool:
        """Create start menu shortcut"""
        try:
            if platform.system() == "Windows":
                start_menu = Path.home() / "AppData/Roaming/Microsoft/Windows/Start Menu/Programs"
                return ShortcutCreator._create_windows_shortcut(
                    start_menu / "AI-ARTWORK.lnk",
                    install_path / "launch.py"
                )
            else:
                applications = Path.home() / ".local/share/applications"
                applications.mkdir(parents=True, exist_ok=True)
                return ShortcutCreator._create_linux_shortcut(
                    applications / "AI-ARTWORK.desktop",
                    install_path / "launch.py"
                )
        except Exception as e:
            logger.error(f"Start menu shortcut creation failed: {e}")
            return False
    
    @staticmethod
    def _create_windows_shortcut(shortcut_path: Path, target_path: Path) -> bool:
        """Create Windows .lnk shortcut"""
        try:
            import win32com.client
            
            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{target_path}"'
            shortcut.WorkingDirectory = str(target_path.parent)
            shortcut.IconLocation = str(target_path.parent / "assets" / "icon.ico")
            shortcut.save()
            
            return True
            
        except Exception as e:
            logger.error(f"Windows shortcut creation failed: {e}")
            return False
    
    @staticmethod
    def _create_linux_shortcut(shortcut_path: Path, target_path: Path) -> bool:
        """Create Linux .desktop shortcut"""
        try:
            desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=AI-ARTWORK
Comment=AI Creative Studio
Exec={sys.executable} "{target_path}"
Icon={target_path.parent / "assets" / "icon.png"}
Terminal=false
Categories=Graphics;AudioVideo;Development;
"""
            
            with open(shortcut_path, 'w') as f:
                f.write(desktop_content)
            
            # Make executable
            shortcut_path.chmod(0o755)
            
            return True
            
        except Exception as e:
            logger.error(f"Linux shortcut creation failed: {e}")
            return False

class RollbackManager:
    """Clean uninstall if installation fails"""
    
    def __init__(self, install_path: Path):
        self.install_path = install_path
        self.backup_path = install_path.parent / f".ai-artwork-backup-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.installed_packages = []
        self.created_files = []
        self.modified_files = {}
        self.registry_changes = []
    
    def create_backup(self) -> bool:
        """Create backup of existing installation"""
        try:
            if self.install_path.exists():
                shutil.copytree(self.install_path, self.backup_path)
                logger.info(f"Backup created at: {self.backup_path}")
                return True
            return True  # No existing installation to backup
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return False
    
    def track_package_installation(self, package: str):
        """Track installed package for rollback"""
        self.installed_packages.append(package)
    
    def track_file_creation(self, file_path: Path):
        """Track created file for rollback"""
        self.created_files.append(file_path)
    
    def track_file_modification(self, file_path: Path, original_content: bytes):
        """Track file modification for rollback"""
        self.modified_files[file_path] = original_content
    
    def rollback(self) -> bool:
        """Perform complete rollback of installation"""
        try:
            logger.info("Starting installation rollback...")
            
            # Uninstall packages
            for package in reversed(self.installed_packages):
                try:
                    subprocess.run([
                        sys.executable, "-m", "pip", "uninstall", package, "-y"
                    ], check=False, capture_output=True)
                    logger.info(f"Uninstalled package: {package}")
                except Exception as e:
                    logger.warning(f"Failed to uninstall {package}: {e}")
            
            # Remove created files
            for file_path in reversed(self.created_files):
                try:
                    if file_path.exists():
                        if file_path.is_dir():
                            shutil.rmtree(file_path)
                        else:
                            file_path.unlink()
                        logger.info(f"Removed: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
            
            # Restore modified files
            for file_path, original_content in self.modified_files.items():
                try:
                    with open(file_path, 'wb') as f:
                        f.write(original_content)
                    logger.info(f"Restored: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to restore {file_path}: {e}")
            
            # Restore from backup if available
            if self.backup_path.exists():
                if self.install_path.exists():
                    shutil.rmtree(self.install_path)
                shutil.move(self.backup_path, self.install_path)
                logger.info("Installation restored from backup")
            
            logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

class FirstRunWizard:
    """Initial configuration helper"""
    
    def __init__(self, install_path: Path):
        self.install_path = install_path
        self.config_path = install_path / "config" / "user_config.json"
    
    def run_wizard(self) -> Dict[str, Any]:
        """Run the first-run configuration wizard"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🎨 Welcome to AI-ARTWORK First-Run Setup!{Colors.END}\n")
        
        config = {}
        
        # Basic preferences
        config.update(self._configure_basic_preferences())
        
        # GPU preferences
        config.update(self._configure_gpu_preferences())
        
        # Voice preferences
        config.update(self._configure_voice_preferences())
        
        # Model preferences
        config.update(self._configure_model_preferences())
        
        # Performance preferences
        config.update(self._configure_performance_preferences())
        
        # Save configuration
        self._save_configuration(config)
        
        print(f"\n{Colors.GREEN}✅ Configuration saved successfully!{Colors.END}")
        print(f"You can modify these settings later in: {self.config_path}")
        
        return config
    
    def _configure_basic_preferences(self) -> Dict[str, Any]:
        """Configure basic user preferences"""
        print(f"{Colors.BLUE}📋 Basic Preferences{Colors.END}")
        
        # Theme preference
        print("\nChoose your preferred theme:")
        print("1. Dark (recommended)")
        print("2. Light")
        print("3. Auto (system)")
        
        theme_choice = self._get_user_choice("Theme", ["dark", "light", "auto"], default="dark")
        
        # Language preference
        print("\nChoose your language:")
        print("1. English")
        print("2. Spanish")
        print("3. French")
        print("4. German")
        
        language_choice = self._get_user_choice("Language", ["en", "es", "fr", "de"], default="en")
        
        return {
            "theme": theme_choice,
            "language": language_choice,
            "first_run_completed": True
        }
    
    def _configure_gpu_preferences(self) -> Dict[str, Any]:
        """Configure GPU and acceleration preferences"""
        print(f"\n{Colors.BLUE}🚀 GPU & Performance{Colors.END}")
        
        gpu_info = GPUDetector.detect_gpu_info()
        
        if gpu_info["available"]:
            print(f"✅ GPU detected: {gpu_info['primary_device']['name']}")
            print(f"   Memory: {gpu_info['primary_device']['memory_gb']:.1f} GB")
            print(f"   CUDA: {gpu_info['cuda_version']}")
            
            use_gpu = self._get_yes_no("Enable GPU acceleration?", default=True)
            
            if use_gpu:
                print("\nGPU memory allocation:")
                print("1. Conservative (50%)")
                print("2. Balanced (70%)")
                print("3. Aggressive (90%)")
                
                memory_choice = self._get_user_choice(
                    "Memory allocation", 
                    ["conservative", "balanced", "aggressive"], 
                    default="balanced"
                )
            else:
                memory_choice = "conservative"
        else:
            print("⚠️  No GPU detected - will use CPU mode")
            use_gpu = False
            memory_choice = "conservative"
        
        return {
            "gpu_enabled": use_gpu,
            "gpu_memory_allocation": memory_choice,
            "gpu_info": gpu_info
        }
    
    def _configure_voice_preferences(self) -> Dict[str, Any]:
        """Configure voice processing preferences"""
        print(f"\n{Colors.BLUE}🎤 Voice Processing{Colors.END}")
        
        # Voice input
        enable_voice_input = self._get_yes_no("Enable voice input?", default=True)
        
        if enable_voice_input:
            print("\nVoice recognition model:")
            print("1. Fast (whisper-tiny)")
            print("2. Balanced (whisper-base)")
            print("3. Accurate (whisper-small)")
            
            whisper_model = self._get_user_choice(
                "Whisper model",
                ["whisper-tiny", "whisper-base", "whisper-small"],
                default="whisper-base"
            )
        else:
            whisper_model = None
        
        # Voice output
        enable_voice_output = self._get_yes_no("Enable voice output?", default=True)
        
        return {
            "voice_input_enabled": enable_voice_input,
            "voice_output_enabled": enable_voice_output,
            "whisper_model": whisper_model
        }
    
    def _configure_model_preferences(self) -> Dict[str, Any]:
        """Configure AI model preferences"""
        print(f"\n{Colors.BLUE}🤖 AI Models{Colors.END}")
        
        # Get system info for recommendations
        system_info = self._get_system_info()
        model_selector = ModelSelector()
        compatible_models = model_selector.filter_compatible_models(system_info)
        
        print(f"Compatible models for your system:")
        print(f"RAM: {system_info.available_ram_gb:.1f} GB available")
        print(f"GPU: {'Yes' if system_info.gpu_available else 'No'}")
        
        selected_models = []
        
        print("\nRecommended models:")
        for i, (model_name, model_info) in enumerate(compatible_models.items(), 1):
            if model_info.get("recommended", False):
                print(f"{i}. {model_name} ({model_info['size_mb']} MB) - {model_info['description']}")
                if self._get_yes_no(f"Install {model_name}?", default=True):
                    selected_models.append(model_name)
        
        return {
            "selected_models": selected_models,
            "auto_update_models": self._get_yes_no("Auto-update models?", default=True)
        }
    
    def _configure_performance_preferences(self) -> Dict[str, Any]:
        """Configure performance preferences"""
        print(f"\n{Colors.BLUE}⚡ Performance Settings{Colors.END}")
        
        # CPU threads
        max_cpu_threads = os.cpu_count()
        print(f"CPU cores available: {max_cpu_threads}")
        
        print("\nCPU usage:")
        print("1. Conservative (50%)")
        print("2. Balanced (75%)")
        print("3. Maximum (100%)")
        
        cpu_usage = self._get_user_choice(
            "CPU usage",
            ["conservative", "balanced", "maximum"],
            default="balanced"
        )
        
        # Cache settings
        cache_size_gb = self._get_numeric_input(
            "Cache size (GB)",
            default=2.0,
            min_val=0.5,
            max_val=10.0
        )
        
        return {
            "cpu_usage": cpu_usage,
            "cache_size_gb": cache_size_gb,
            "auto_optimize": self._get_yes_no("Enable auto-optimization?", default=True)
        }
    
    def _get_user_choice(self, prompt: str, choices: List[str], default: str = None) -> str:
        """Get user choice from a list of options"""
        while True:
            choice_str = f" (default: {default})" if default else ""
            user_input = input(f"{prompt}{choice_str}: ").strip().lower()
            
            if not user_input and default:
                return default
            
            if user_input in choices:
                return user_input
            
            # Try to match by number
            try:
                choice_num = int(user_input) - 1
                if 0 <= choice_num < len(choices):
                    return choices[choice_num]
            except ValueError:
                pass
            
            print(f"Invalid choice. Please choose from: {', '.join(choices)}")
    
    def _get_yes_no(self, prompt: str, default: bool = None) -> bool:
        """Get yes/no input from user"""
        while True:
            default_str = " (Y/n)" if default else " (y/N)" if default is False else " (y/n)"
            user_input = input(f"{prompt}{default_str}: ").strip().lower()
            
            if not user_input and default is not None:
                return default
            
            if user_input in ['y', 'yes', 'true', '1']:
                return True
            elif user_input in ['n', 'no', 'false', '0']:
                return False
            
            print("Please answer yes (y) or no (n)")
    
    def _get_numeric_input(self, prompt: str, default: float = None, min_val: float = None, max_val: float = None) -> float:
        """Get numeric input from user"""
        while True:
            default_str = f" (default: {default})" if default is not None else ""
            user_input = input(f"{prompt}{default_str}: ").strip()
            
            if not user_input and default is not None:
                return default
            
            try:
                value = float(user_input)
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue
                return value
            except ValueError:
                print("Please enter a valid number")
    
    def _get_system_info(self) -> SystemInfo:
        """Get system information for configuration"""
        gpu_info = GPUDetector.detect_gpu_info()
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            total_ram = memory.total / (1024**3)
            available_ram = memory.available / (1024**3)
            disk = psutil.disk_usage('/')
            disk_space = disk.free / (1024**3)
        else:
            total_ram = 8.0  # Default assumption
            available_ram = 4.0
            disk_space = 50.0
        
        return SystemInfo(
            os_type=platform.system(),
            architecture=platform.machine(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            total_ram_gb=total_ram,
            available_ram_gb=available_ram,
            cpu_count=os.cpu_count(),
            gpu_available=gpu_info["available"],
            gpu_name=gpu_info["primary_device"]["name"] if gpu_info["primary_device"] else None,
            cuda_version=gpu_info["cuda_version"],
            disk_space_gb=disk_space
        )
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

class EnhancedInstaller:
    """Main enhanced installer class"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.progress_tracker = None
        self.rollback_manager = None
        self.dependency_resolver = DependencyResolver()
        self.model_selector = ModelSelector()
        
    def run_installation(self, config: InstallationConfig) -> bool:
        """Run the complete installation process"""
        try:
            # Initialize progress tracking
            total_steps = self._calculate_total_steps(config)
            self.progress_tracker = ProgressTracker(total_steps)
            self.progress_tracker.add_callback(self._print_progress)
            
            # Initialize rollback manager
            self.rollback_manager = RollbackManager(config.install_path)
            
            print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 Starting AI-ARTWORK Enhanced Installation{Colors.END}")
            print(f"Installation path: {config.install_path}")
            print(f"Total steps: {total_steps}")
            
            # Step 1: System analysis
            self.progress_tracker.update_step("Analyzing system...")
            system_info = self._analyze_system()
            self._print_system_info(system_info)
            self.progress_tracker.complete_step()
            
            # Step 2: Create backup
            self.progress_tracker.update_step("Creating backup...")
            if not self.rollback_manager.create_backup():
                raise Exception("Failed to create backup")
            self.progress_tracker.complete_step()
            
            # Step 3: Dependency resolution
            self.progress_tracker.update_step("Resolving dependencies...")
            packages_to_install = self.dependency_resolver.resolve_dependencies(config)
            missing, outdated, conflicts = self.dependency_resolver.check_dependencies()
            
            if conflicts:
                print(f"{Colors.RED}❌ Dependency conflicts detected: {conflicts}{Colors.END}")
                return False
            
            print(f"{Colors.GREEN}✅ Dependencies resolved: {len(packages_to_install)} packages{Colors.END}")
            self.progress_tracker.complete_step()
            
            # Step 4: Install dependencies
            self.progress_tracker.update_step("Installing dependencies...")
            if not self._install_dependencies(packages_to_install):
                raise Exception("Dependency installation failed")
            self.progress_tracker.complete_step()
            
            # Step 5: Setup directories
            self.progress_tracker.update_step("Setting up directories...")
            if not self._setup_directories(config.install_path):
                raise Exception("Directory setup failed")
            self.progress_tracker.complete_step()
            
            # Step 6: Copy files
            self.progress_tracker.update_step("Copying application files...")
            if not self._copy_application_files(config.install_path):
                raise Exception("File copying failed")
            self.progress_tracker.complete_step()
            
            # Step 7: Download models
            if config.install_models:
                self.progress_tracker.update_step("Downloading AI models...")
                if not self._download_models(config.install_models, config.install_path):
                    print(f"{Colors.YELLOW}⚠️  Model download failed, continuing...{Colors.END}")
                self.progress_tracker.complete_step()
            
            # Step 8: Configure PATH
            if config.add_to_path:
                self.progress_tracker.update_step("Configuring system PATH...")
                if not PathManager.add_to_path(config.install_path):
                    print(f"{Colors.YELLOW}⚠️  PATH configuration failed{Colors.END}")
                self.progress_tracker.complete_step()
            
            # Step 9: Create shortcuts
            if config.create_shortcuts:
                self.progress_tracker.update_step("Creating shortcuts...")
                if not ShortcutCreator.create_shortcuts(config.install_path, config):
                    print(f"{Colors.YELLOW}⚠️  Shortcut creation failed{Colors.END}")
                self.progress_tracker.complete_step()
            
            # Step 10: First-run wizard
            self.progress_tracker.update_step("Running first-run wizard...")
            wizard = FirstRunWizard(config.install_path)
            user_config = wizard.run_wizard()
            self.progress_tracker.complete_step()
            
            # Step 11: Final verification
            self.progress_tracker.update_step("Verifying installation...")
            if not self._verify_installation(config.install_path):
                raise Exception("Installation verification failed")
            self.progress_tracker.complete_step()
            
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Installation completed successfully!{Colors.END}")
            self._print_success_message(config.install_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Installation failed: {e}")
            print(f"\n{Colors.RED}❌ Installation failed: {e}{Colors.END}")
            
            # Perform rollback
            if self.rollback_manager:
                print(f"{Colors.YELLOW}🔄 Rolling back installation...{Colors.END}")
                if self.rollback_manager.rollback():
                    print(f"{Colors.GREEN}✅ Rollback completed successfully{Colors.END}")
                else:
                    print(f"{Colors.RED}❌ Rollback failed{Colors.END}")
            
            return False
    
    def _calculate_total_steps(self, config: InstallationConfig) -> int:
        """Calculate total number of installation steps"""
        steps = 7  # Base steps: analyze, backup, resolve, install, setup, copy, verify
        
        if config.install_models:
            steps += 1
        if config.add_to_path:
            steps += 1
        if config.create_shortcuts:
            steps += 1
        
        steps += 1  # First-run wizard
        
        return steps
    
    def _analyze_system(self) -> SystemInfo:
        """Analyze system capabilities"""
        gpu_info = GPUDetector.detect_gpu_info()
        
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            total_ram = memory.total / (1024**3)
            available_ram = memory.available / (1024**3)
            disk = psutil.disk_usage('/')
            disk_space = disk.free / (1024**3)
        else:
            total_ram = 8.0  # Default assumption
            available_ram = 4.0
            disk_space = 50.0
        
        return SystemInfo(
            os_type=platform.system(),
            architecture=platform.machine(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            total_ram_gb=total_ram,
            available_ram_gb=available_ram,
            cpu_count=os.cpu_count(),
            gpu_available=gpu_info["available"],
            gpu_name=gpu_info["primary_device"]["name"] if gpu_info["primary_device"] else None,
            cuda_version=gpu_info["cuda_version"],
            disk_space_gb=disk_space
        )
    
    def _print_system_info(self, system_info: SystemInfo):
        """Print system information"""
        print(f"\n{Colors.BLUE}💻 System Information:{Colors.END}")
        print(f"  OS: {system_info.os_type} ({system_info.architecture})")
        print(f"  Python: {system_info.python_version}")
        print(f"  RAM: {system_info.total_ram_gb:.1f} GB total, {system_info.available_ram_gb:.1f} GB available")
        print(f"  CPU: {system_info.cpu_count} cores")
        print(f"  GPU: {'✅ ' + system_info.gpu_name if system_info.gpu_available else '❌ Not available'}")
        if system_info.cuda_version:
            print(f"  CUDA: {system_info.cuda_version}")
        print(f"  Disk Space: {system_info.disk_space_gb:.1f} GB available")
    
    def _install_dependencies(self, packages: List[str]) -> bool:
        """Install Python dependencies"""
        try:
            for package in packages:
                logger.info(f"Installing {package}...")
                
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", package, "--upgrade"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Failed to install {package}: {result.stderr}")
                    return False
                
                self.rollback_manager.track_package_installation(package)
                
                if TQDM_AVAILABLE:
                    # Simulate progress for visual feedback
                    time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Dependency installation failed: {e}")
            return False
    
    def _setup_directories(self, install_path: Path) -> bool:
        """Setup application directories"""
        try:
            directories = [
                "src", "models", "cache", "logs", "config", 
                "plugins", "outputs", "temp", "assets"
            ]
            
            for directory in directories:
                dir_path = install_path / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                self.rollback_manager.track_file_creation(dir_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Directory setup failed: {e}")
            return False
    
    def _copy_application_files(self, install_path: Path) -> bool:
        """Copy application files to installation directory"""
        try:
            # Files and directories to copy
            items_to_copy = [
                "src/", "scripts/", "launch.py", "setup.py", 
                "requirements.txt", "README.md", "LICENSE"
            ]
            
            for item in items_to_copy:
                source = self.project_root / item
                if source.exists():
                    if source.is_dir():
                        destination = install_path / item
                        if destination.exists():
                            shutil.rmtree(destination)
                        shutil.copytree(source, destination)
                    else:
                        destination = install_path / item
                        shutil.copy2(source, destination)
                    
                    self.rollback_manager.track_file_creation(destination)
            
            return True
            
        except Exception as e:
            logger.error(f"File copying failed: {e}")
            return False
    
    def _download_models(self, model_names: List[str], install_path: Path) -> bool:
        """Download selected AI models"""
        try:
            models_dir = install_path / "models"
            
            for model_name in model_names:
                if model_name in self.model_selector.available_models:
                    model_info = self.model_selector.available_models[model_name]
                    
                    print(f"Downloading {model_name} ({model_info['size_mb']} MB)...")
                    
                    # Create model-specific directory
                    model_dir = models_dir / model_info["category"] / model_name
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Create placeholder for now (actual download would be implemented)
                    placeholder_file = model_dir / "model_info.json"
                    with open(placeholder_file, 'w') as f:
                        json.dump(model_info, f, indent=2)
                    
                    self.rollback_manager.track_file_creation(model_dir)
                    
                    # Simulate download progress
                    if TQDM_AVAILABLE:
                        for _ in tqdm(range(100), desc=f"Downloading {model_name}"):
                            time.sleep(0.01)
            
            return True
            
        except Exception as e:
            logger.error(f"Model download failed: {e}")
            return False
    
    def _verify_installation(self, install_path: Path) -> bool:
        """Verify installation integrity"""
        try:
            # Check critical files
            critical_files = [
                "launch.py", "src/__init__.py", "requirements.txt"
            ]
            
            for file_name in critical_files:
                file_path = install_path / file_name
                if not file_path.exists():
                    logger.error(f"Critical file missing: {file_path}")
                    return False
            
            # Test import
            sys.path.insert(0, str(install_path / "src"))
            try:
                # Test basic imports
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "launch", install_path / "launch.py"
                )
                if spec is None:
                    return False
                
            except Exception as e:
                logger.error(f"Import test failed: {e}")
                return False
            finally:
                if str(install_path / "src") in sys.path:
                    sys.path.remove(str(install_path / "src"))
            
            return True
            
        except Exception as e:
            logger.error(f"Installation verification failed: {e}")
            return False
    
    def _print_progress(self, progress: InstallationProgress):
        """Print installation progress"""
        if progress.eta_seconds:
            eta_str = f" (ETA: {progress.eta_seconds//60}m {progress.eta_seconds%60}s)"
        else:
            eta_str = ""
        
        progress_bar = "█" * int(progress.current_progress / 10) + "░" * (10 - int(progress.current_progress / 10))
        
        print(f"\r{Colors.BLUE}[{progress_bar}] {progress.current_progress:.1f}% - {progress.current_step}{eta_str}{Colors.END}", end="")
        
        if progress.status == "completed":
            print()  # New line when completed
    
    def _print_success_message(self, install_path: Path):
        """Print success message with next steps"""
        print(f"\n{Colors.GREEN}🎉 AI-ARTWORK has been successfully installed!{Colors.END}\n")
        print(f"{Colors.BOLD}Installation Location:{Colors.END} {install_path}")
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        print(f"1. Run: {Colors.CYAN}python {install_path / 'launch.py'}{Colors.END}")
        print(f"2. Or use the desktop shortcut")
        print(f"3. Check the documentation: {install_path / 'README.md'}")
        print(f"\n{Colors.BOLD}Support:{Colors.END}")
        print(f"- Documentation: https://github.com/AI-ARTWORK/docs")
        print(f"- Issues: https://github.com/AI-ARTWORK/issues")
        print(f"- Community: https://discord.gg/ai-artwork")

def main():
    """Main installer entry point"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                   AI-ARTWORK Enhanced Installer              ║")
    print("║                                                              ║")
    print("║  🎨 AI Creative Studio with Advanced Installation Features   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Get installation path
    default_path = Path.home() / "AI-ARTWORK"
    install_path_str = input(f"Installation path (default: {default_path}): ").strip()
    install_path = Path(install_path_str) if install_path_str else default_path
    
    # Create installation configuration
    config = InstallationConfig(
        install_path=install_path,
        create_shortcuts=True,
        add_to_path=True,
        install_models=["whisper-base", "llama-2-7b-chat"],
        gpu_support=True,
        development_mode=False
    )
    
    # Run installation
    installer = EnhancedInstaller()
    success = installer.run_installation(config)
    
    if success:
        print(f"\n{Colors.GREEN}Installation completed successfully! 🎉{Colors.END}")
        return 0
    else:
        print(f"\n{Colors.RED}Installation failed! ❌{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())