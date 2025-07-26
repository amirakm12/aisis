# How to Write a Plugin for AI-ARTWORK

AI-ARTWORK supports modular plugins for extending functionality. Here's how to create your own:

## 1. Inherit from PluginBase
```python
from plugins.plugin_base import PluginBase

class MyPlugin(PluginBase):
    def run(self, *args, **kwargs):
        # Your plugin logic here
        pass
```

## 2. Implement the run() Method
- Accept any arguments you need.
- Return results as a dict or value.

## 3. Add Docstrings and Type Hints
- Document your plugin for users and contributors.

## 4. Test Your Plugin
- Place your plugin in the `plugins/` folder.
- Use the PluginManager to load and run it.

## 5. Example: Batch Image Processor
```python
from plugins.plugin_base import PluginBase
from pathlib import Path
from PIL import Image

class BatchProcessorPlugin(PluginBase):
    def run(self, input_dir: str, agent, output_dir: str = "output", **kwargs):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for img_file in input_path.glob("*.jpg"):
            image = Image.open(img_file)
            result = agent.process({'image': image, **kwargs})
            if isinstance(result, dict) and 'output_image' in result:
                result['output_image'].save(output_path / img_file.name)
```

## 6. Best Practices
- Use type hints and docstrings.
- Handle errors gracefully.
- Avoid global state.
- Document any dependencies.

---
See `plugins/README.md` for more details and examples. 