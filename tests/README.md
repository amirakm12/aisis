# AISIS Test Suite

This folder contains all tests for AISIS, covering agents, core modules, UI, and integration.

## Structure
- `test_agents.py`: Tests for agent modules and workflows.
- `test_core.py`: Tests for core utilities and infrastructure.
- `test_integration.py`: End-to-end and integration tests.
- `test_performance.py`: Performance and stress tests.
- `test_ui.py`: UI and interaction tests.

## Running Tests
Run all tests with pytest from the project root:

```bash
pytest
```

Or run a specific test file:

```bash
pytest tests/test_agents.py
```

## Guidelines
- Use pytest for all new tests.
- Add clear docstrings and comments.
- Aim for high coverage, especially for core and agent logic.
- Add regression tests for all bug fixes.

---
See each test file for detailed coverage and usage examples. 