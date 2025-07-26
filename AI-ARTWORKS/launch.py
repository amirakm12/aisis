#!/usr/bin/env python3
"""
AI-ARTWORK Launch Script
Sets up environment and launches the AI Creative Studio
"""

import os
import sys
import asyncio
from pathlib import Path
from loguru import logger

def setup_environment():
    """Setup Python path and environment variables"""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Add src to Python path
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set environment variables
    os.environ["AI_ARTWORK_ROOT"] = str(project_root)
    os.environ["PYTHONPATH"] = f"{src_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    
    # Create necessary directories
    directories = ["models", "cache", "logs", "outputs", "temp"]
    for directory in directories:
        (project_root / directory).mkdir(exist_ok=True)
    
    logger.info(f"AI-ARTWORK environment setup complete")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Python path includes: {src_path}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "torch", "torchvision", "PIL", "numpy", "loguru",
        "PySide6", "whisper", "soundfile"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed")
    return True

async def launch_gui():
    """Launch the GUI version of AI-ARTWORK"""
    try:
        from src.ui.main_window import MainWindow
        from PySide6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        
        logger.info("AI-ARTWORK GUI launched successfully")
        return app.exec()
        
    except Exception as e:
        logger.error(f"Failed to launch GUI: {e}")
        return 1

async def launch_cli():
    """Launch the CLI version of AI-ARTWORK"""
    try:
        from src import AI_ARTWORK
        
        studio = AI_ARTWORK()
        await studio.initialize()
        
        logger.info("AI-ARTWORK CLI launched successfully")
        logger.info("Available commands:")
        logger.info("  - edit_image(image_path, instruction)")
        logger.info("  - generate_image(prompt)")
        logger.info("  - reconstruct_3d(image_path)")
        
        # Interactive CLI loop
        while True:
            try:
                command = input("\nAI-ARTWORK> ").strip()
                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.startswith('edit '):
                    # Parse edit command: edit <image_path> <instruction>
                    parts = command[5:].split(' ', 1)
                    if len(parts) == 2:
                        image_path, instruction = parts
                        result = await studio.edit_image(image_path, instruction)
                        logger.info(f"Edit result: {result}")
                elif command.startswith('generate '):
                    # Parse generate command: generate <prompt>
                    prompt = command[9:]
                    result = await studio.generate_image(prompt)
                    logger.info(f"Generation result: {result}")
                elif command.startswith('3d '):
                    # Parse 3D command: 3d <image_path>
                    image_path = command[3:]
                    result = await studio.reconstruct_3d(image_path)
                    logger.info(f"3D reconstruction result: {result}")
                else:
                    logger.info("Unknown command. Use: edit, generate, 3d, or quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")
        
        await studio.cleanup()
        return 0
        
    except Exception as e:
        logger.error(f"Failed to launch CLI: {e}")
        return 1

def main():
    """Main launch function"""
    # Setup logging
    logger.add(
        "logs/ai-artwork_launch.log",
        rotation="10 MB",
        level="INFO"
    )
    
    logger.info("Starting AI-ARTWORK...")
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "gui"  # Default to GUI
    
    # Launch appropriate mode
    if mode == "cli":
        return asyncio.run(launch_cli())
    elif mode == "gui":
        return asyncio.run(launch_gui())
    else:
        logger.error(f"Unknown mode: {mode}. Use 'gui' or 'cli'")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 