# AISIS Agents

This folder contains all specialized AI agents for the AISIS Creative Studio. Each agent is a modular, extensible Python class responsible for a specific image processing or analysis task.

## Architecture
- All agents inherit from `BaseAgent` and implement a standard interface.
- Agents are orchestrated by the HyperOrchestrator for complex, multi-step workflows.
- New agents can be added by subclassing `BaseAgent` and registering with the orchestrator.

## Agent Types
| Agent Name                | Purpose                                      |
|--------------------------|----------------------------------------------|
| image_restoration.py      | Core restoration (inpainting, denoising, etc)|
| style_aesthetic.py        | Style transfer and aesthetic enhancement     |
| semantic_editing.py       | Context-aware, language-driven editing       |
| auto_retouch.py           | Face/body detection and enhancement          |
| generative.py             | Diffusion/generative models (SDXL, etc.)     |
| style_transfer.py         | Neural style transfer and artistic transformation |
| vision_language.py        | Vision-language tasks (captioning, retrieval, etc.) |
| ...                      | ... (see each file for details)              |

## Adding a New Agent
1. Subclass `BaseAgent` and implement required methods (`initialize`, `process`, etc.).
2. Add your agent to the orchestrator in `src/__init__.py`.
3. Document your agent with clear docstrings and usage examples.

## Extension Guidelines
- Use type hints and docstrings for all public methods.
- Handle errors gracefully and log important events.
- Keep agent logic modular and testable.

---
See each agent file for detailed documentation and implementation notes. 