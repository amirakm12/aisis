# Al-artworks UI Modules

This folder contains the graphical user interface (GUI) code for Al-artworks, built with PySide6/Qt.

## UI Components

### Main Interface
- `main_window.py` - Primary application window
- `modern_interface.py` - Modern UI components and themes
- `context_panel.py` - Context-aware interface panels
- `context_manager.py` - Context management system

### Dialogs and Modals
- `tour_dialog.py` - Welcome tour and onboarding
- `onboarding_dialog.py` - User onboarding flow
- `agent_explain_dialog.py` - AI agent explanations
- `crash_reporting_dialog.py` - Error reporting interface
- `plugin_manager_dialog.py` - Plugin management interface

### Interactive Components
- `chat_panel.py` - Chat interface for AI interaction
- `learning_panel.py` - Learning and feedback interface
- `loading_screen.py` - Loading and progress indicators
- `notifications.py` - Notification system

### Settings and Configuration
- `settings_panel.py` - Application settings interface
- `model_zoo_dialog.py` - Model selection and management

## Key Features

### Modern Design
- Dark/light theme support
- Responsive layout
- Smooth animations
- Professional styling

### User Experience
- Intuitive navigation
- Context-aware interfaces
- Real-time feedback
- Accessibility support

### AI Integration
- Natural language chat
- Voice input/output
- Visual feedback
- Progress indicators

### Customization
- Theme switching
- Layout customization
- Plugin support
- User preferences

## Architecture

The UI follows a modular architecture:

```
UI Layer
├── Main Window
│   ├── Toolbar
│   ├── Sidebar
│   ├── Content Area
│   └── Status Bar
├── Dialogs
│   ├── Settings
│   ├── Onboarding
│   └── Notifications
├── Panels
│   ├── Chat
│   ├── Learning
│   └── Context
└── Components
    ├── Buttons
    ├── Inputs
    └── Displays
```

## Usage Examples

### Creating the Main Window
```python
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

app = QApplication([])
window = MainWindow()
window.show()
app.exec()
```

### Adding a Custom Dialog
```python
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel

class CustomDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Custom Dialog")
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Hello from Al-artworks!"))
```

### Theme Management
```python
from src.ui.modern_interface import ModernThemeManager

# Apply dark theme
theme_manager = ModernThemeManager()
theme_manager.apply_theme(app, "dark")

# Apply light theme
theme_manager.apply_theme(app, "light")
```

### Chat Integration
```python
from src.ui.chat_panel import ChatPanel

# Create chat panel
chat = ChatPanel()

# Send message
chat.send_message("Hello, AI!")

# Handle AI response
def on_ai_response(response):
    chat.display_ai_message(response)

chat.ai_response_received.connect(on_ai_response)
```

## Styling

### CSS-like Styling
```python
# Apply custom styles
window.setStyleSheet("""
    QMainWindow {
        background-color: #2a2a2a;
        color: #ffffff;
    }
    
    QPushButton {
        background-color: #4a4a4a;
        border: 1px solid #666;
        padding: 8px;
        border-radius: 4px;
    }
    
    QPushButton:hover {
        background-color: #5a5a5a;
    }
""")
```

### Theme Variables
```python
# Define theme colors
DARK_THEME = {
    "background": "#2a2a2a",
    "foreground": "#ffffff",
    "accent": "#007acc",
    "error": "#ff4444",
    "success": "#44ff44"
}
```

## Event Handling

### Signal Connections
```python
# Connect button click
button.clicked.connect(self.on_button_clicked)

# Connect custom signal
self.data_changed.connect(self.update_display)

# Connect with lambda
button.clicked.connect(lambda: self.process_data(data))
```

### Custom Events
```python
from PySide6.QtCore import QEvent

class CustomEvent(QEvent):
    EVENT_TYPE = QEvent.Type(QEvent.registerUserEvent())
    
    def __init__(self, data):
        super().__init__(self.EVENT_TYPE)
        self.data = data

# Post custom event
app.postEvent(widget, CustomEvent("custom data"))
```

## Performance

### Optimization Tips
- Use QThread for long-running operations
- Implement virtual scrolling for large lists
- Cache frequently accessed data
- Minimize widget updates
- Use QTimer for periodic updates

### Memory Management
- Properly dispose of widgets
- Use weak references where appropriate
- Avoid circular references
- Clean up resources in destructors

## Testing

### UI Testing
```python
import pytest
from PySide6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

@pytest.fixture
def app():
    return QApplication([])

@pytest.fixture
def window(app):
    return MainWindow()

def test_window_creation(window):
    assert window.windowTitle() == "Al-artworks - AI Creative Studio"
    assert window.isVisible() == False
```

### Automated Testing
- Use pytest-qt for Qt testing
- Test user interactions
- Verify visual feedback
- Check accessibility features

## Dependencies

- **PySide6** - Qt bindings for Python
- **Qt** - Cross-platform GUI framework
- **Pillow** - Image processing
- **NumPy** - Numerical operations

## Development Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Keep methods small and focused

### UI Design
- Follow Qt design patterns
- Use consistent naming
- Implement proper error handling
- Add loading states

### Accessibility
- Include keyboard shortcuts
- Add screen reader support
- Use proper ARIA labels
- Test with accessibility tools 