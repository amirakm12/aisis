"""
AISIS Plugin Manager
Handles loading, managing, and executing plugins
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
import json
import shutil
from dataclasses import asdict
from loguru import logger

from .base_plugin import BasePlugin, PluginMetadata
from .sandbox import PluginSandbox
from src.core.config import config

class PluginManager:
    """
    Manages all AISIS plugins including loading, enabling, disabling, and execution
    """
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_metadata: Dict[str, PluginMetadata] = {}
        self.enabled_plugins: Dict[str, bool] = {}
        self.plugin_paths: Dict[str, Path] = {}
        self.sandbox = PluginSandbox()
        
        # Plugin directories
        self.system_plugin_dir = Path(__file__).parent.parent.parent / "plugins"
        self.user_plugin_dir = Path.home() / ".aisis" / "plugins"
        
        # Ensure directories exist
        self.system_plugin_dir.mkdir(exist_ok=True)
        self.user_plugin_dir.mkdir(parents=True, exist_ok=True)
        
        # Load plugin registry
        self.registry_file = self.user_plugin_dir / "registry.json"
        self.load_registry()
    
    def load_registry(self):
        """Load plugin registry from file"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    registry_data = json.load(f)
                    self.enabled_plugins = registry_data.get('enabled', {})
            except Exception as e:
                logger.warning(f"Failed to load plugin registry: {e}")
                self.enabled_plugins = {}
        else:
            self.enabled_plugins = {}
    
    def save_registry(self):
        """Save plugin registry to file"""
        try:
            registry_data = {
                'enabled': self.enabled_plugins,
                'metadata': {
                    name: asdict(metadata) 
                    for name, metadata in self.plugin_metadata.items()
                }
            }
            
            with open(self.registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save plugin registry: {e}")
    
    def discover_plugins(self) -> List[Path]:
        """Discover all available plugins"""
        plugin_files = []
        
        # Search system plugins
        for plugin_file in self.system_plugin_dir.rglob("*.py"):
            if plugin_file.name != "__init__.py":
                plugin_files.append(plugin_file)
        
        # Search user plugins
        for plugin_file in self.user_plugin_dir.rglob("*.py"):
            if plugin_file.name != "__init__.py":
                plugin_files.append(plugin_file)
        
        return plugin_files
    
    def load_plugin(self, plugin_path: Path) -> Optional[BasePlugin]:
        """Load a single plugin from file"""
        try:
            # Create module spec
            module_name = f"aisis_plugin_{plugin_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            
            if spec is None or spec.loader is None:
                logger.error(f"Failed to create spec for plugin: {plugin_path}")
                return None
            
            # Load module
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find plugin class
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BasePlugin) and 
                    attr != BasePlugin):
                    plugin_class = attr
                    break
            
            if plugin_class is None:
                logger.error(f"No plugin class found in: {plugin_path}")
                return None
            
            # Create plugin instance
            plugin = plugin_class()
            
            # Validate plugin
            if not self.validate_plugin(plugin):
                logger.error(f"Plugin validation failed: {plugin_path}")
                return None
            
            # Store plugin info
            plugin_name = plugin.get_metadata().name
            self.plugins[plugin_name] = plugin
            self.plugin_metadata[plugin_name] = plugin.get_metadata()
            self.plugin_paths[plugin_name] = plugin_path
            
            # Set default enabled state
            if plugin_name not in self.enabled_plugins:
                self.enabled_plugins[plugin_name] = True
            
            logger.info(f"Loaded plugin: {plugin_name}")
            return plugin
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return None
    
    def validate_plugin(self, plugin: BasePlugin) -> bool:
        """Validate a plugin meets requirements"""
        try:
            metadata = plugin.get_metadata()
            
            # Check required fields
            if not metadata.name or not metadata.version:
                return False
            
            # Check AISIS version compatibility
            if hasattr(plugin, 'min_aisis_version'):
                # Would check version compatibility here
                pass
            
            # Check if plugin has required methods
            required_methods = ['initialize', 'execute', 'cleanup']
            for method in required_methods:
                if not hasattr(plugin, method):
                    logger.error(f"Plugin missing required method: {method}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin validation error: {e}")
            return False
    
    def load_plugins(self):
        """Load all discovered plugins"""
        logger.info("Loading AISIS plugins...")
        
        plugin_files = self.discover_plugins()
        loaded_count = 0
        
        for plugin_file in plugin_files:
            if self.load_plugin(plugin_file):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} plugins")
        
        # Initialize enabled plugins
        self.initialize_enabled_plugins()
        
        # Save registry
        self.save_registry()
    
    def initialize_enabled_plugins(self):
        """Initialize all enabled plugins"""
        for plugin_name, plugin in self.plugins.items():
            if self.enabled_plugins.get(plugin_name, False):
                try:
                    plugin.initialize()
                    logger.info(f"Initialized plugin: {plugin_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
                    self.enabled_plugins[plugin_name] = False
    
    def enable_plugin(self, plugin_name: str):
        """Enable a plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if not self.enabled_plugins.get(plugin_name, False):
            try:
                self.plugins[plugin_name].initialize()
                self.enabled_plugins[plugin_name] = True
                self.save_registry()
                logger.info(f"Enabled plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to enable plugin {plugin_name}: {e}")
                raise
    
    def disable_plugin(self, plugin_name: str):
        """Disable a plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if self.enabled_plugins.get(plugin_name, False):
            try:
                self.plugins[plugin_name].cleanup()
                self.enabled_plugins[plugin_name] = False
                self.save_registry()
                logger.info(f"Disabled plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to disable plugin {plugin_name}: {e}")
                raise
    
    def uninstall_plugin(self, plugin_name: str):
        """Uninstall a plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        # Disable first
        if self.enabled_plugins.get(plugin_name, False):
            self.disable_plugin(plugin_name)
        
        # Remove plugin files
        plugin_path = self.plugin_paths[plugin_name]
        if plugin_path.exists():
            if plugin_path.is_file():
                plugin_path.unlink()
            else:
                shutil.rmtree(plugin_path)
        
        # Remove from memory
        del self.plugins[plugin_name]
        del self.plugin_metadata[plugin_name]
        del self.plugin_paths[plugin_name]
        del self.enabled_plugins[plugin_name]
        
        self.save_registry()
        logger.info(f"Uninstalled plugin: {plugin_name}")
    
    def install_plugin(self, source: str):
        """Install a plugin from file path or URL"""
        source_path = Path(source)
        
        if source_path.exists():
            # Install from local file
            dest_path = self.user_plugin_dir / source_path.name
            shutil.copy2(source_path, dest_path)
            
            # Load the plugin
            plugin = self.load_plugin(dest_path)
            if plugin:
                logger.info(f"Installed plugin from: {source}")
            else:
                dest_path.unlink()  # Remove if failed to load
                raise ValueError("Failed to load plugin after installation")
        else:
            # Could implement URL download here
            raise NotImplementedError("URL installation not yet implemented")
    
    def execute_plugin(self, plugin_name: str, *args, **kwargs) -> Any:
        """Execute a plugin safely"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        if not self.enabled_plugins.get(plugin_name, False):
            raise ValueError(f"Plugin not enabled: {plugin_name}")
        
        plugin = self.plugins[plugin_name]
        
        try:
            # Execute in sandbox for security
            return self.sandbox.execute_plugin(plugin, *args, **kwargs)
        except Exception as e:
            logger.error(f"Plugin execution failed {plugin_name}: {e}")
            raise
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their status"""
        plugins_info = []
        
        for plugin_name, plugin in self.plugins.items():
            metadata = self.plugin_metadata[plugin_name]
            plugins_info.append({
                'name': plugin_name,
                'version': metadata.version,
                'description': metadata.description,
                'author': metadata.author,
                'enabled': self.enabled_plugins.get(plugin_name, False),
                'capabilities': metadata.tags,
                'path': str(self.plugin_paths[plugin_name])
            })
        
        return plugins_info
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin instance"""
        return self.plugins.get(plugin_name)
    
    def get_enabled_plugins(self) -> Dict[str, BasePlugin]:
        """Get all enabled plugins"""
        return {
            name: plugin for name, plugin in self.plugins.items()
            if self.enabled_plugins.get(name, False)
        }
    
    def unload_all_plugins(self):
        """Unload all plugins and cleanup"""
        for plugin_name, plugin in self.plugins.items():
            if self.enabled_plugins.get(plugin_name, False):
                try:
                    plugin.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up plugin {plugin_name}: {e}")
        
        self.plugins.clear()
        self.plugin_metadata.clear()
        self.enabled_plugins.clear()
        self.plugin_paths.clear()
        
        logger.info("All plugins unloaded")
    
    def reload_plugin(self, plugin_name: str):
        """Reload a specific plugin"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin not found: {plugin_name}")
        
        # Get plugin path
        plugin_path = self.plugin_paths[plugin_name]
        was_enabled = self.enabled_plugins.get(plugin_name, False)
        
        # Disable and remove
        if was_enabled:
            self.disable_plugin(plugin_name)
        
        # Remove from memory
        del self.plugins[plugin_name]
        del self.plugin_metadata[plugin_name]
        
        # Reload
        plugin = self.load_plugin(plugin_path)
        if plugin and was_enabled:
            self.enable_plugin(plugin_name)
        
        logger.info(f"Reloaded plugin: {plugin_name}")

# Global plugin manager instance
plugin_manager = PluginManager()