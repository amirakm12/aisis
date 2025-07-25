#!/usr/bin/env python3
"""
AISIS Health Check Script
Comprehensive validation of project structure and dependencies
"""

import os
import sys
import importlib
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

class AISISHealthChecker:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.warnings = []
        self.suggestions = []
        
    def log_issue(self, category: str, message: str, severity: str = "error"):
        """Log an issue with the project"""
        issue = {
            "category": category,
            "message": message,
            "severity": severity
        }
        
        if severity == "error":
            self.issues.append(issue)
        elif severity == "warning":
            self.warnings.append(issue)
        else:
            self.suggestions.append(issue)
    
    def check_file_structure(self) -> bool:
        """Check if all required files and directories exist"""
        print("🔍 Checking project file structure...")
        
        required_files = [
            "main.py",
            "launch.py", 
            "requirements.txt",
            "config.json",
            "setup.py",
            "pyproject.toml",
            "README.md",
            "LICENSE"
        ]
        
        required_dirs = [
            "src",
            "src/core",
            "src/ui", 
            "src/agents",
            "src/voice",
            "src/plugins",
            "tests",
            "scripts",
            "docs"
        ]
        
        all_good = True
        
        # Check required files
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.log_issue("file_structure", f"Missing required file: {file_path}")
                all_good = False
            else:
                print(f"✅ {file_path}")
        
        # Check required directories
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                self.log_issue("file_structure", f"Missing required directory: {dir_path}")
                all_good = False
            else:
                print(f"✅ {dir_path}/")
        
        return all_good
    
    def check_python_imports(self) -> bool:
        """Check if all required Python modules can be imported"""
        print("\n🐍 Checking Python imports...")
        
        # Core Python modules that should be available
        core_modules = [
            "sys", "os", "json", "pathlib", "asyncio", "subprocess",
            "typing", "logging", "datetime", "collections"
        ]
        
        # Required third-party modules
        required_modules = [
            "numpy", "PIL", "cv2", "torch", "torchvision", "torchaudio",
            "loguru", "pydantic", "requests", "psutil", "tqdm"
        ]
        
        # Optional but recommended modules
        optional_modules = [
            ("PySide6", "GUI framework"),
            ("whisper", "Speech recognition"),
            ("transformers", "Transformer models"),
            ("diffusers", "Diffusion models"),
            ("accelerate", "Model acceleration"),
            ("cryptography", "Security features")
        ]
        
        all_good = True
        
        # Test core modules
        for module in core_modules:
            try:
                importlib.import_module(module)
                print(f"✅ {module}")
            except ImportError:
                self.log_issue("imports", f"Missing core Python module: {module}")
                all_good = False
        
        # Test required modules
        for module in required_modules:
            try:
                importlib.import_module(module)
                print(f"✅ {module}")
            except ImportError:
                self.log_issue("imports", f"Missing required module: {module}")
                all_good = False
        
        # Test optional modules
        for module, description in optional_modules:
            try:
                importlib.import_module(module)
                print(f"✅ {module} ({description})")
            except ImportError:
                self.log_issue("imports", f"Missing optional module: {module} - {description}", "warning")
        
        return all_good
    
    def check_project_modules(self) -> bool:
        """Check if project-specific modules can be imported"""
        print("\n📦 Checking project modules...")
        
        # Add project root to Python path
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
        
        project_modules = [
            "src.core.config",
            "src.core.device", 
            "src.core.model_manager",
            "src.ui.main_window",
            "src.agents.base_agent",
            "src.voice.voice_manager"
        ]
        
        all_good = True
        
        for module in project_modules:
            try:
                importlib.import_module(module)
                print(f"✅ {module}")
            except ImportError as e:
                self.log_issue("project_modules", f"Cannot import {module}: {e}")
                all_good = False
            except Exception as e:
                self.log_issue("project_modules", f"Error in {module}: {e}", "warning")
        
        return all_good
    
    def check_configuration(self) -> bool:
        """Check configuration files"""
        print("\n⚙️  Checking configuration...")
        
        config_file = self.project_root / "config.json"
        all_good = True
        
        if not config_file.exists():
            self.log_issue("config", "config.json file missing")
            return False
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            required_sections = ["ui", "voice", "llm", "gpu", "paths"]
            for section in required_sections:
                if section not in config_data:
                    self.log_issue("config", f"Missing configuration section: {section}")
                    all_good = False
                else:
                    print(f"✅ Config section: {section}")
        
        except json.JSONDecodeError as e:
            self.log_issue("config", f"Invalid JSON in config.json: {e}")
            all_good = False
        
        return all_good
    
    def check_virtual_environment(self) -> bool:
        """Check if virtual environment is properly set up"""
        print("\n🌐 Checking virtual environment...")
        
        venv_path = self.project_root / "venv"
        
        if not venv_path.exists():
            self.log_issue("venv", "Virtual environment not found", "warning")
            return False
        
        # Check if we're running in the virtual environment
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            print("✅ Running in virtual environment")
            return True
        else:
            self.log_issue("venv", "Not running in virtual environment", "warning")
            return False
    
    def check_gpu_availability(self) -> bool:
        """Check GPU availability for AI processing"""
        print("\n🖥️  Checking GPU availability...")
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                gpu_name = torch.cuda.get_device_name(current_device)
                print(f"✅ CUDA available: {gpu_count} GPU(s)")
                print(f"✅ Current GPU: {gpu_name}")
                return True
            else:
                self.log_issue("gpu", "CUDA not available - will use CPU only", "warning")
                return False
        except ImportError:
            self.log_issue("gpu", "PyTorch not available - cannot check GPU", "warning")
            return False
    
    def check_model_directories(self) -> bool:
        """Check if model directories are set up"""
        print("\n🤖 Checking model directories...")
        
        models_dir = Path.home() / ".aisis" / "models"
        
        if not models_dir.exists():
            self.log_issue("models", "Models directory not found", "warning")
            return False
        
        # Check for essential models
        essential_models = ["whisper-base", "clip-vit-base"]
        models_found = 0
        
        for model in essential_models:
            model_path = models_dir / model
            if model_path.exists():
                print(f"✅ Model found: {model}")
                models_found += 1
            else:
                self.log_issue("models", f"Missing model: {model}", "warning")
        
        return models_found > 0
    
    def check_permissions(self) -> bool:
        """Check file permissions"""
        print("\n🔐 Checking permissions...")
        
        # Check if we can write to the project directory
        test_file = self.project_root / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print("✅ Project directory writable")
        except PermissionError:
            self.log_issue("permissions", "Cannot write to project directory")
            return False
        
        # Check if we can create directories in home
        home_test_dir = Path.home() / ".aisis_test"
        try:
            home_test_dir.mkdir(exist_ok=True)
            home_test_dir.rmdir()
            print("✅ Home directory writable")
        except PermissionError:
            self.log_issue("permissions", "Cannot write to home directory")
            return False
        
        return True
    
    def suggest_fixes(self):
        """Suggest fixes for identified issues"""
        print("\n🔧 Suggested fixes:")
        
        if self.issues or self.warnings:
            print("\n❌ Issues found:")
            for issue in self.issues:
                print(f"   ERROR [{issue['category']}]: {issue['message']}")
            
            for warning in self.warnings:
                print(f"   WARNING [{warning['category']}]: {warning['message']}")
        
        # Provide specific fix suggestions
        fix_suggestions = {
            "imports": "Run: python install.py to install missing dependencies",
            "file_structure": "Ensure all required files are present in the project",
            "venv": "Create virtual environment: python -m venv venv",
            "config": "Check config.json syntax and required sections",
            "models": "Run: python scripts/download_models.py",
            "gpu": "Install CUDA drivers and PyTorch with CUDA support",
            "permissions": "Check file/directory permissions"
        }
        
        categories_with_issues = set()
        for issue in self.issues + self.warnings:
            categories_with_issues.add(issue['category'])
        
        if categories_with_issues:
            print("\n💡 Fix suggestions:")
            for category in categories_with_issues:
                if category in fix_suggestions:
                    print(f"   {category}: {fix_suggestions[category]}")
    
    def run_health_check(self) -> bool:
        """Run complete health check"""
        print("🏥 AISIS Health Check Starting...")
        print("=" * 50)
        
        checks = [
            ("File Structure", self.check_file_structure),
            ("Python Imports", self.check_python_imports),
            ("Project Modules", self.check_project_modules),
            ("Configuration", self.check_configuration),
            ("Virtual Environment", self.check_virtual_environment),
            ("GPU Availability", self.check_gpu_availability),
            ("Model Directories", self.check_model_directories),
            ("Permissions", self.check_permissions),
        ]
        
        results = []
        for check_name, check_func in checks:
            print(f"\n📋 {check_name}:")
            result = check_func()
            results.append(result)
        
        # Summary
        print("\n" + "=" * 50)
        passed_checks = sum(results)
        total_checks = len(results)
        
        if passed_checks == total_checks and not self.issues:
            print("🎉 All health checks passed!")
            print("✅ AISIS is ready to run")
            return True
        else:
            print(f"⚠️  Health check results: {passed_checks}/{total_checks} passed")
            print(f"❌ {len(self.issues)} errors, {len(self.warnings)} warnings")
            self.suggest_fixes()
            return False

def main():
    """Main health check function"""
    checker = AISISHealthChecker()
    
    if "--fix" in sys.argv:
        print("🔧 Auto-fix mode not yet implemented")
        print("Please run the suggested fixes manually")
    
    success = checker.run_health_check()
    
    if success:
        print("\n🚀 Ready to launch AISIS!")
        print("Run: python main.py")
    else:
        print("\n🛠️  Please fix the issues above before running AISIS")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()