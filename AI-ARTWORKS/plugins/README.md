# AI-ARTWORK Plugin System

This folder contains the plugin system for AI-ARTWORK, enabling modular extensions and third-party integrations.

## Architecture
- All plugins inherit from `PluginBase` and implement a `run()` method.
- Plugins are discovered and loaded by `PluginManager` at runtime.
- Plugins can access a limited context for sandboxing and security.

## Example Plugin: BatchProcessorPlugin
This plugin applies a given agent to all images in a folder for batch processing.

## Example Plugin: ImageCaptionExporterPlugin
This plugin uses a vision-language agent to caption all images in a folder and export results to a CSV file.

## Adding a New Plugin
1. Subclass `PluginBase` and implement the `run()` method.
2. Place your plugin in this folder (as a `.py` file or package).
3. Register any required dependencies in the plugin docstring.

## Guidelines
- Use type hints and docstrings for all public methods.
- Avoid accessing global state or sensitive data.
- Document plugin functionality and requirements.

## Security Notes
- Plugins run in the same process as the main app. Only install trusted plugins.
- Future versions may add stricter sandboxing or plugin signing.

---
See `plugin_base.py` and `plugin_manager.py` for implementation details. 