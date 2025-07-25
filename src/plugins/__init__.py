"""
AISIS Plugin System
Extensible plugin architecture for AISIS
"""

from .plugin_manager import PluginManager
from .base_plugin import BasePlugin, PluginMetadata
from .plugin_loader import PluginLoader
from .plugin_registry import PluginRegistry

# Create global plugin manager instance
plugin_manager = PluginManager()

__all__ = [
    'PluginManager',
    'BasePlugin',
    'PluginMetadata',
    'PluginLoader',
    'PluginRegistry',
    'plugin_manager'
] 