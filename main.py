"""
AISIS - AI Creative Studio
Main application entry point
"""

import sys
from pathlib import Path
import asyncio
from loguru import logger

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QCoreApplication

from src.ui import MainWindow
from src.core.config import config

def setup_logging():
    """Setup logging configuration"""
    log_path = Path.home() / ".aisis" / "logs" / "aisis.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_path,
        rotation="10 MB",
        retention="1 week",
        level="INFO"
    )

def main():
    """Main application entry point"""
    # Setup logging
    setup_logging()
    logger.info("Starting AISIS")
    
    # Enable high DPI scaling
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("AISIS")
    app.setOrganizationName("AISIS")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Enable OpenGL
    try:
        from PySide6.QtGui import QSurfaceFormat
        surface_format = QSurfaceFormat()
        surface_format.setRenderableType(QSurfaceFormat.OpenGLES)
        surface_format.setVersion(3, 0)
        QSurfaceFormat.setDefaultFormat(surface_format)
    except Exception as e:
        logger.warning(f"Failed to set OpenGL format: {e}")
    
    # Create and show main window
    try:
        window = MainWindow()
        window.show()
        
        # Start event loop
        return app.exec()
        
    except Exception as e:
        logger.error(f"Failed to start AISIS: {e}")
        return 1
    
    finally:
        logger.info("AISIS shutdown complete")

if __name__ == "__main__":
    sys.exit(main())
