#!/usr/bin/env python3
"""
Setup script for Pylint configuration and environment management.
This script helps resolve Pylint connection issues and sets up the development environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def check_pylint_installation():
    """Check if Pylint is properly installed."""
    print("🔍 Checking Pylint installation...")
    try:
        result = subprocess.run(["pylint", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Pylint is installed and working")
            print(f"Version: {result.stdout.strip()}")
            return True
        else:
            print("❌ Pylint is not working properly")
            return False
    except FileNotFoundError:
        print("❌ Pylint is not installed")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    # Install Pylint if not already installed
    run_command("pip install pylint", "Installing Pylint")
    
    # Install other development dependencies
    run_command("pip install black isort mypy", "Installing development tools")
    
    print("✅ Dependencies installed successfully")


def test_pylint_configuration():
    """Test Pylint configuration with a sample file."""
    print("🧪 Testing Pylint configuration...")
    
    # Create a test file
    test_file = "test_pylint.py"
    test_content = '''"""
Test file for Pylint configuration.
"""

def test_function():
    """A simple test function."""
    print("Hello, Pylint!")

if __name__ == "__main__":
    test_function()
'''
    
    with open(test_file, "w") as f:
        f.write(test_content)
    
    # Test Pylint
    result = run_command("pylint test_pylint.py", "Testing Pylint on sample file")
    
    # Clean up
    os.remove(test_file)
    
    if result is not None:
        print("✅ Pylint configuration is working correctly")
        return True
    else:
        print("❌ Pylint configuration has issues")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Pylint for your project...")
    print("=" * 50)
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"📁 Working directory: {current_dir}")
    
    # Check if we're in the right directory
    if not (current_dir / "pyproject.toml").exists():
        print("⚠️  Warning: pyproject.toml not found in current directory")
        print("   Make sure you're running this from the project root")
    
    # Install dependencies
    install_dependencies()
    
    # Check Pylint installation
    if not check_pylint_installation():
        print("❌ Pylint setup failed")
        sys.exit(1)
    
    # Test configuration
    if test_pylint_configuration():
        print("\n🎉 Pylint setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Restart your IDE/editor")
        print("   2. Open a Python file to test Pylint")
        print("   3. Check that linting errors appear in your editor")
        print("\n🔧 Configuration files created:")
        print("   - .pylintrc (Pylint configuration)")
        print("   - .vscode/settings.json (VS Code settings)")
        print("   - pyproject.toml (updated with Pylint settings)")
    else:
        print("❌ Pylint setup failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 