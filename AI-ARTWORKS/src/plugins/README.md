# Al-artworks Plugin System

This folder contains the plugin system for Al-artworks, providing extensible functionality through modular plugins.

## Plugin Architecture

### Core Components
- `extension_api.py` - Plugin API and interface definitions
- `sandbox.py` - Secure plugin execution environment
- `__init__.py` - Plugin system initialization

### Plugin Types
- **Image Processing Plugins** - Custom image filters and effects
- **AI Model Plugins** - Additional AI models and algorithms
- **UI Extension Plugins** - Custom interface components
- **Integration Plugins** - External service connections
- **Utility Plugins** - Helper tools and utilities

## Plugin Development

### Creating a Plugin
```python
from src.plugins.extension_api import Plugin, PluginMetadata

class MyPlugin(Plugin):
    """Example plugin for Al-artworks"""
    
    def __init__(self):
        self.metadata = PluginMetadata(
            name="My Plugin",
            version="1.0.0",
            author="Your Name",
            description="A sample plugin",
            category="image_processing"
        )
    
    async def initialize(self):
        """Initialize the plugin"""
        print("Plugin initialized")
    
    async def process(self, data):
        """Process input data"""
        # Your plugin logic here
        return {"status": "success", "result": data}
    
    async def cleanup(self):
        """Cleanup plugin resources"""
        print("Plugin cleaned up")
```

### Plugin Registration
```python
from src.plugins import register_plugin

# Register your plugin
register_plugin(MyPlugin())
```

### Plugin Configuration
```python
class ConfigurablePlugin(Plugin):
    def __init__(self, config=None):
        self.config = config or {}
        self.metadata = PluginMetadata(
            name="Configurable Plugin",
            version="1.0.0",
            config_schema={
                "param1": {"type": "string", "default": "value"},
                "param2": {"type": "int", "default": 10}
            }
        )
    
    async def process(self, data):
        param1 = self.config.get("param1", "default")
        param2 = self.config.get("param2", 10)
        
        # Use configuration parameters
        return {"status": "success", "param1": param1, "param2": param2}
```

## Plugin Categories

### Image Processing Plugins
```python
class ImageFilterPlugin(Plugin):
    """Plugin for image filtering"""
    
    def __init__(self):
        self.metadata = PluginMetadata(
            name="Image Filter",
            category="image_processing",
            input_types=["image"],
            output_types=["image"]
        )
    
    async def process(self, data):
        image = data.get("image")
        if image:
            # Apply filter to image
            filtered_image = self.apply_filter(image)
            return {"status": "success", "image": filtered_image}
        
        return {"status": "error", "message": "No image provided"}
    
    def apply_filter(self, image):
        # Your filter implementation
        return image
```

### AI Model Plugins
```python
class CustomAIModelPlugin(Plugin):
    """Plugin for custom AI models"""
    
    def __init__(self):
        self.metadata = PluginMetadata(
            name="Custom AI Model",
            category="ai_model",
            model_path="models/custom_model.pth"
        )
        self.model = None
    
    async def initialize(self):
        """Load the AI model"""
        self.model = self.load_model()
    
    async def process(self, data):
        if self.model:
            result = self.model.predict(data)
            return {"status": "success", "prediction": result}
        
        return {"status": "error", "message": "Model not loaded"}
```

### UI Extension Plugins
```python
class CustomUIPlugin(Plugin):
    """Plugin for UI extensions"""
    
    def __init__(self):
        self.metadata = PluginMetadata(
            name="Custom UI",
            category="ui_extension"
        )
    
    def create_widget(self):
        """Create custom UI widget"""
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
        
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        button = QPushButton("Custom Action")
        button.clicked.connect(self.custom_action)
        layout.addWidget(button)
        
        return widget
    
    def custom_action(self):
        """Handle custom UI action"""
        print("Custom action triggered")
```

## Plugin Management

### Loading Plugins
```python
from src.plugins import PluginManager

# Initialize plugin manager
plugin_manager = PluginManager()

# Load plugins from directory
await plugin_manager.load_plugins("plugins/")

# Get available plugins
plugins = plugin_manager.get_plugins()

# Execute plugin
result = await plugin_manager.execute_plugin("my_plugin", data)
```

### Plugin Discovery
```python
# Auto-discover plugins
discovered_plugins = plugin_manager.discover_plugins()

# Load specific plugin
plugin = plugin_manager.load_plugin("path/to/plugin.py")

# Enable/disable plugins
plugin_manager.enable_plugin("my_plugin")
plugin_manager.disable_plugin("my_plugin")
```

## Security

### Sandboxed Execution
```python
from src.plugins.sandbox import SandboxedPlugin

class SafePlugin(SandboxedPlugin):
    """Plugin running in sandboxed environment"""
    
    def __init__(self):
        super().__init__()
        self.restricted_imports = ["os", "subprocess"]
        self.allowed_functions = ["print", "len"]
    
    async def process(self, data):
        # Safe execution environment
        return {"status": "success", "data": data}
```

### Security Guidelines
- Validate all plugin inputs
- Restrict file system access
- Limit network access
- Monitor resource usage
- Implement timeout mechanisms

## Testing Plugins

### Unit Testing
```python
import pytest
from src.plugins import MyPlugin

@pytest.mark.asyncio
async def test_plugin():
    plugin = MyPlugin()
    await plugin.initialize()
    
    result = await plugin.process({"test": "data"})
    assert result["status"] == "success"
    
    await plugin.cleanup()
```

### Integration Testing
```python
@pytest.mark.asyncio
async def test_plugin_integration():
    plugin_manager = PluginManager()
    await plugin_manager.load_plugin(MyPlugin())
    
    result = await plugin_manager.execute_plugin("MyPlugin", {"test": "data"})
    assert result["status"] == "success"
```

## Plugin Distribution

### Plugin Package Structure
```
my_plugin/
├── __init__.py
├── plugin.py
├── requirements.txt
├── README.md
└── tests/
    └── test_plugin.py
```

### Plugin Installation
```python
# Install plugin from package
plugin_manager.install_plugin("path/to/plugin_package")

# Install from repository
plugin_manager.install_from_repo("https://github.com/user/plugin")

# Update plugin
plugin_manager.update_plugin("my_plugin")
```

## Best Practices

### Plugin Development
- Follow the plugin interface contract
- Implement proper error handling
- Use async/await for I/O operations
- Add comprehensive documentation
- Include unit tests

### Performance
- Optimize for speed and memory usage
- Implement caching where appropriate
- Use efficient algorithms
- Monitor resource consumption

### Compatibility
- Test with different Al-artworks versions
- Handle backward compatibility
- Use stable APIs
- Document dependencies

## Troubleshooting

### Common Issues
1. **Plugin not loading** - Check file permissions and syntax
2. **Import errors** - Verify dependencies are installed
3. **Performance issues** - Profile and optimize code
4. **Security violations** - Review sandbox configuration

### Debug Mode
```python
# Enable plugin debugging
import logging
logging.getLogger('src.plugins').setLevel(logging.DEBUG)

# Verbose plugin execution
plugin_manager.set_verbose(True)
``` 