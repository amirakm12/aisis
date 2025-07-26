# AI-ARTWORK Plugin Infrastructure

## Overview
AI-ARTWORK supports a flexible plugin system for extending functionality.

## Writing a Plugin
- Inherit from `plugin_base.PluginBase` in `plugins/plugin_base.py`.
- Implement required methods (e.g., `run`, `get_metadata`).
- Place your plugin in the `plugins/` directory.

## Plugin Manager
- The plugin manager (`plugins/plugin_manager.py`) loads, enables, and disables plugins.
- Use the UI or CLI to manage plugins.

## Example
```python
from plugins.plugin_base import PluginBase

class MyPlugin(PluginBase):
    def run(self, *args, **kwargs):
        # Your plugin logic here
        pass
```

## More Information
- See `docs/HowToWriteAPlugin.md` for a step-by-step guide.
- Explore sample plugins in the `plugins/` directory. 