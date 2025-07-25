"""
Base Plugin Class for AISIS
All plugins must inherit from this class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class PluginMetadata:
    """Plugin metadata structure"""
    name: str
    version: str
    description: str
    author: str
    license: str = "MIT"
    homepage: Optional[str] = None
    tags: List[str] = None
    dependencies: List[str] = None
    min_aisis_version: str = "1.0.0"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []

class BasePlugin(ABC):
    """
    Base class for all AISIS plugins
    
    All plugins must inherit from this class and implement the required methods.
    """
    
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.metadata = self._load_metadata()
        self.config = {}
        self.enabled = False
        self._hooks = {}
    
    def _load_metadata(self) -> PluginMetadata:
        """Load plugin metadata from plugin.json"""
        metadata_file = self.plugin_dir / "plugin.json"
        
        if not metadata_file.exists():
            raise ValueError(f"Plugin metadata file not found: {metadata_file}")
        
        try:
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            return PluginMetadata(**data)
        except Exception as e:
            raise ValueError(f"Invalid plugin metadata: {e}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the plugin
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """
        Cleanup plugin resources
        Called when plugin is disabled or AISIS shuts down
        """
        pass
    
    def configure(self, config: Dict[str, Any]):
        """
        Configure the plugin with settings
        
        Args:
            config: Plugin configuration dictionary
        """
        self.config = config
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get plugin information
        
        Returns:
            Dict containing plugin metadata and status
        """
        return {
            "name": self.metadata.name,
            "version": self.metadata.version,
            "description": self.metadata.description,
            "author": self.metadata.author,
            "license": self.metadata.license,
            "homepage": self.metadata.homepage,
            "tags": self.metadata.tags,
            "enabled": self.enabled,
            "plugin_dir": str(self.plugin_dir)
        }
    
    def register_hook(self, hook_name: str, callback):
        """
        Register a hook callback
        
        Args:
            hook_name: Name of the hook
            callback: Function to call when hook is triggered
        """
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
    
    def trigger_hook(self, hook_name: str, *args, **kwargs):
        """
        Trigger a hook
        
        Args:
            hook_name: Name of the hook to trigger
            *args: Arguments to pass to hook callbacks
            **kwargs: Keyword arguments to pass to hook callbacks
        """
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    print(f"Error in hook {hook_name}: {e}")
    
    def enable(self) -> bool:
        """
        Enable the plugin
        
        Returns:
            bool: True if enabled successfully
        """
        if not self.enabled:
            try:
                if self.initialize():
                    self.enabled = True
                    return True
            except Exception as e:
                print(f"Failed to enable plugin {self.metadata.name}: {e}")
        return False
    
    def disable(self):
        """Disable the plugin"""
        if self.enabled:
            try:
                self.cleanup()
                self.enabled = False
            except Exception as e:
                print(f"Error disabling plugin {self.metadata.name}: {e}")
    
    def get_assets_path(self) -> Path:
        """Get path to plugin assets directory"""
        return self.plugin_dir / "assets"
    
    def get_config_path(self) -> Path:
        """Get path to plugin config file"""
        return self.plugin_dir / "config.json"
    
    def save_config(self):
        """Save plugin configuration to file"""
        config_path = self.get_config_path()
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Failed to save plugin config: {e}")
    
    def load_config(self):
        """Load plugin configuration from file"""
        config_path = self.get_config_path()
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Failed to load plugin config: {e}")

class ImageProcessingPlugin(BasePlugin):
    """Base class for image processing plugins"""
    
    @abstractmethod
    def process_image(self, image_path: Path, **kwargs) -> Path:
        """
        Process an image
        
        Args:
            image_path: Path to input image
            **kwargs: Additional processing parameters
            
        Returns:
            Path to processed image
        """
        pass

class AgentPlugin(BasePlugin):
    """Base class for AI agent plugins"""
    
    @abstractmethod
    def create_agent(self, config: Dict[str, Any]):
        """
        Create an AI agent instance
        
        Args:
            config: Agent configuration
            
        Returns:
            Agent instance
        """
        pass

class UIPlugin(BasePlugin):
    """Base class for UI extension plugins"""
    
    @abstractmethod
    def create_widget(self, parent=None):
        """
        Create UI widget
        
        Args:
            parent: Parent widget
            
        Returns:
            Widget instance
        """
        pass
    
    @abstractmethod
    def get_menu_items(self) -> List[Dict[str, Any]]:
        """
        Get menu items to add to the main menu
        
        Returns:
            List of menu item dictionaries
        """
        pass