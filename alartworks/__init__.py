"""
Al-artworks - AI Creative Studio
Main package providing comprehensive AI-powered creative tools
"""

__version__ = "1.0.0"
__author__ = "Al-artworks Team"
__license__ = "MIT"
__description__ = "AI-powered creative studio with advanced image processing and restoration capabilities"

# Core imports
from src.core.config import config
from src.core.model_manager import ModelManager
from src.core.device import DeviceManager
from src.agents.multi_agent_orchestrator import MultiAgentOrchestrator

# UI imports
try:
    from src.ui.main_window import MainWindow
    from src.ui.modern_interface import ModernInterface
    UI_AVAILABLE = True
except ImportError:
    UI_AVAILABLE = False

# Plugin system
from src.plugins.plugin_manager import PluginManager

# Main API classes
__all__ = [
    'AlArtworks',
    'config',
    'ModelManager',
    'DeviceManager',
    'MultiAgentOrchestrator',
    'PluginManager',
    '__version__'
]

class AlArtworks:
    """
    Main Al-artworks class providing the primary API interface
    """
    
    def __init__(self):
        self.config = config
        self.model_manager = ModelManager()
        self.device_manager = DeviceManager()
        self.orchestrator = MultiAgentOrchestrator()
        self.plugin_manager = PluginManager()
        self._initialized = False
    
    def initialize(self):
        """Initialize all Al-artworks components"""
        if self._initialized:
            return
            
        # Initialize core components
        self.device_manager.initialize()
        self.model_manager.initialize()
        self.orchestrator.initialize()
        self.plugin_manager.load_plugins()
        
        self._initialized = True
    
    def create_gui(self):
        """Create and return the main GUI window"""
        if not UI_AVAILABLE:
            raise RuntimeError("UI components not available. Install PySide6.")
        
        if not self._initialized:
            self.initialize()
            
        return MainWindow()
    
    def process_image(self, image_path, operations=None):
        """
        Process an image using the agent orchestrator
        
        Args:
            image_path: Path to the image file
            operations: List of operations to perform
            
        Returns:
            Processed image result
        """
        if not self._initialized:
            self.initialize()
            
        return self.orchestrator.process_image(image_path, operations)
    
    def get_available_agents(self):
        """Get list of available agents"""
        if not self._initialized:
            self.initialize()
            
        return self.orchestrator.get_available_agents()
    
    def shutdown(self):
        """Clean shutdown of all components"""
        if hasattr(self, 'plugin_manager'):
            self.plugin_manager.unload_all_plugins()
        if hasattr(self, 'model_manager'):
            self.model_manager.cleanup()
        self._initialized = False

# Global Al-artworks instance
alartworks = AlArtworks()