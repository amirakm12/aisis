#!/usr/bin/env python3
"""
AISIS Comprehensive Launcher
Main entry point for running AISIS in different modes
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'PySide6', 'torch', 'torchvision', 'numpy', 'Pillow',
        'loguru', 'click', 'fastapi', 'uvicorn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        
        response = input("\nWould you like to install missing packages? (y/n): ")
        if response.lower() in ['y', 'yes']:
            install_dependencies(missing_packages)
        else:
            print("Cannot continue without required packages.")
            sys.exit(1)

def install_dependencies(packages: List[str]):
    """Install missing dependencies"""
    print("Installing missing packages...")
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            sys.exit(1)
    
    print("All packages installed successfully!")

def run_gui():
    """Launch the AISIS GUI"""
    print("Starting AISIS GUI...")
    try:
        from aisis import aisis
        aisis.initialize()
        
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
        
        window = aisis.create_gui()
        window.show()
        
        return app.exec()
        
    except Exception as e:
        print(f"Failed to start GUI: {e}")
        return 1

def run_cli(args: List[str]):
    """Run AISIS CLI commands"""
    try:
        from aisis.cli import cli
        
        # Remove the 'cli' argument if present
        if args and args[0] == 'cli':
            args = args[1:]
        
        cli(args, standalone_mode=False)
        
    except SystemExit as e:
        return e.code
    except Exception as e:
        print(f"CLI error: {e}")
        return 1

def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the AISIS API server"""
    print(f"Starting AISIS API server on {host}:{port}")
    try:
        import uvicorn
        from aisis.api import app
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False
        )
        
    except Exception as e:
        print(f"Failed to start API server: {e}")
        return 1

def run_tests():
    """Run the test suite"""
    print("Running AISIS test suite...")
    try:
        import pytest
        return pytest.main([
            "tests/test_comprehensive.py",
            "-v",
            "--tb=short"
        ])
    except ImportError:
        print("pytest not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
        return pytest.main([
            "tests/test_comprehensive.py", 
            "-v",
            "--tb=short"
        ])

def run_health_check():
    """Run health check"""
    print("Running AISIS health check...")
    try:
        from health_check import AISISHealthChecker
        
        checker = AISISHealthChecker()
        report = checker.run_full_check()
        
        print("\n" + "="*50)
        print("AISIS HEALTH CHECK REPORT")
        print("="*50)
        
        if report.get('issues'):
            print("\n❌ ISSUES FOUND:")
            for issue in report['issues']:
                print(f"  - {issue['message']}")
        
        if report.get('warnings'):
            print("\n⚠️  WARNINGS:")
            for warning in report['warnings']:
                print(f"  - {warning['message']}")
        
        if report.get('suggestions'):
            print("\n💡 SUGGESTIONS:")
            for suggestion in report['suggestions']:
                print(f"  - {suggestion}")
        
        if not report.get('issues') and not report.get('warnings'):
            print("\n✅ All systems operational!")
        
        return 0
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return 1

def run_install():
    """Run installation script"""
    print("Running AISIS installation...")
    try:
        from install import AISISInstaller
        
        installer = AISISInstaller()
        
        # Check system requirements
        if not installer.check_system_requirements():
            print("System requirements not met")
            return 1
        
        # Install dependencies
        installer.install_dependencies()
        
        # Setup environment
        installer.setup_environment()
        
        print("✅ AISIS installation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Installation failed: {e}")
        return 1

def run_benchmark():
    """Run performance benchmarks"""
    print("Running AISIS benchmarks...")
    try:
        from aisis import aisis
        aisis.initialize()
        
        # Device benchmark
        device_info = aisis.device_manager.get_device_info()
        
        print("\n" + "="*40)
        print("DEVICE INFORMATION")
        print("="*40)
        print(f"GPU: {device_info.get('gpu', 'None')}")
        print(f"Memory: {device_info.get('memory', 'Unknown')}")
        print(f"Compute Capability: {device_info.get('compute_capability', 'Unknown')}")
        
        # Model benchmarks
        try:
            from src.core.model_benchmarking import ModelBenchmarker
            benchmarker = ModelBenchmarker()
            results = benchmarker.run_benchmarks()
            
            print("\n" + "="*40)
            print("MODEL PERFORMANCE")
            print("="*40)
            for model_name, metrics in results.items():
                print(f"{model_name}:")
                print(f"  Inference Time: {metrics.get('inference_time', 'N/A')}ms")
                print(f"  Memory Usage: {metrics.get('memory_usage', 'N/A')}MB")
                print(f"  Quality Score: {metrics.get('quality_score', 'N/A')}/100")
        except Exception as e:
            print(f"Model benchmarking failed: {e}")
        
        return 0
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="AISIS - AI Creative Studio Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_aisis.py gui                    # Launch GUI
  python run_aisis.py cli agents             # List available agents
  python run_aisis.py api                    # Start API server
  python run_aisis.py test                   # Run tests
  python run_aisis.py health                 # Run health check
  python run_aisis.py install                # Run installation
  python run_aisis.py benchmark              # Run benchmarks
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['gui', 'cli', 'api', 'test', 'health', 'install', 'benchmark'],
        help='Mode to run AISIS in'
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='API server host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='API server port (default: 8000)'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency checks'
    )
    
    # Parse known args to allow passing through to CLI
    args, unknown = parser.parse_known_args()
    
    # System checks
    check_python_version()
    
    if not args.skip_checks:
        check_dependencies()
    
    # Run the requested mode
    exit_code = 0
    
    try:
        if args.mode == 'gui':
            exit_code = run_gui()
        elif args.mode == 'cli':
            exit_code = run_cli(unknown)
        elif args.mode == 'api':
            exit_code = run_api(args.host, args.port)
        elif args.mode == 'test':
            exit_code = run_tests()
        elif args.mode == 'health':
            exit_code = run_health_check()
        elif args.mode == 'install':
            exit_code = run_install()
        elif args.mode == 'benchmark':
            exit_code = run_benchmark()
        else:
            print(f"Unknown mode: {args.mode}")
            exit_code = 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        exit_code = 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)

if __name__ == '__main__':
    main()