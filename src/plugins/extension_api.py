import importlib
import os
import sys
from typing import Dict, Type, Any


class ExtensionBase:
    """Base class for all plugins/extensions."""

    def __init__(self, context: Dict[str, Any] = None):
        self.context = context or {}

    def run(self, *args, **kwargs):
        raise NotImplementedError("Plugins must implement the run() method.")


class ExtensionRegistry:
    """Registry for all loaded plugins/extensions."""

    def __init__(self):
        self.plugins: Dict[str, Type[ExtensionBase]] = {}

    def register(self, name: str, plugin: Type[ExtensionBase]):
        self.plugins[name] = plugin

    def get(self, name: str) -> Type[ExtensionBase]:
        return self.plugins.get(name)

    def list(self):
        return list(self.plugins.keys())


registry = ExtensionRegistry()


def discover_plugins(directory: str):
    """Dynamically discover and register plugins in a directory."""
    sys.path.insert(0, directory)
    for fname in os.listdir(directory):
        if fname.endswith(".py") and not fname.startswith("_"):
            modname = fname[:-3]
            mod = importlib.import_module(modname)
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, ExtensionBase)
                    and obj is not ExtensionBase
                ):
                    registry.register(modname, obj)


# Usage for third-party developers:
# 1. Subclass ExtensionBase and implement run().
# 2. Place your plugin in the plugins/ directory.
# 3. Call discover_plugins('plugins') to auto-register.
