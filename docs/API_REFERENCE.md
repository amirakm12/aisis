# AISIS API Reference

## Core Modules
- `src/core/config.py`: Configuration management
- `src/core/device.py`: Device and hardware abstraction
- `src/core/advanced_local_models.py`: Local model management

## Agents
- `src/agents/`: Image restoration, enhancement, orchestration, and more
  - Example: `adaptive_enhancement.py`, `auto_retouch.py`, `super_resolution.py`

## Plugins
- `plugins/`: Plugin system and available plugins
  - Example: `batch_processor.py`, `image_caption_exporter.py`

## UI
- `src/ui/`: User interface components and dialogs
  - Example: `main_window.py`, `context_panel.py`

## Collaboration
- `src/collab/`: Collaboration server
- `src/ui/collab_client.py`: Collaboration client

## Detailed API

### Core Modules

#### config.py
- `load_config()`: Loads configuration from JSON
- Parameters: path (str)
- Returns: dict

#### advanced_local_models.py
- `LocalModelManager`: Manages model loading
- Methods: download_model(model_name), load_model(model_name)

### Agents

#### adaptive_enhancement.py
- `AdaptiveEnhancementAgent`: Handles image enhancement
- Methods: process(image, params)

Add similar details for other modules.

---
For detailed docstrings and function/class documentation, see the source code or use tools like `pydoc` or `help()` in Python. 