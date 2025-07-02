# AISIS UI Modules

This folder contains the graphical user interface (GUI) code for AISIS, built with PySide6/Qt.

## Main Components
- `main_window.py`: The main application window, chat interface, and image display.

## Architecture
- The UI is modular and event-driven, with clear separation of concerns.
- Voice, chat, and image editing are integrated for a seamless user experience.

## Extending the UI
- Add new widgets or dialogs as separate modules.
- Use Qt signals/slots for async operations and UI updates.
- Follow PEP8 and Qt best practices for layout and accessibility.

## Guidelines
- Use type hints and docstrings for all public methods.
- Ensure all UI elements are accessible and have tooltips/help.
- Keep the UI responsive and visually appealing.

---
See `main_window.py` for implementation details and extension points. 