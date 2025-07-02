# AISIS Core Modules

This folder contains the core infrastructure for AISIS, including configuration, GPU utilities, LLM management, and voice management.

## Modules
| Module                | Purpose                                      |
|-----------------------|----------------------------------------------|
| config.py             | Centralized configuration management         |
| gpu_utils.py          | GPU detection, memory management, optimization|
| llm_manager.py        | Local LLM (Llama) management and inference   |
| voice_manager.py      | Voice input/output and streaming ASR         |
| logging_setup.py      | (Optional) Centralized logging configuration |

## Guidelines
- All core modules should be PEP8-compliant and use type hints.
- Add docstrings to all public classes and methods.
- Keep core logic modular and reusable.

---
See each module for detailed documentation and usage examples. 