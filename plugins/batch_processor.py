from .plugin_base import PluginBase
from typing import Any, Dict
from pathlib import Path
from PIL import Image

class BatchProcessorPlugin(PluginBase):
    """
    Plugin for batch processing images in a folder using a specified agent.
    """
    def run(self, input_dir: str, agent, output_dir: str = "output", **kwargs) -> Dict[str, Any]:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        results = {}
        for img_file in input_path.glob("*.jpg"):
            try:
                image = Image.open(img_file)
                result = agent.process({'image': image, **kwargs})
                out_file = output_path / img_file.name
                if isinstance(result, dict) and 'output_image' in result:
                    result['output_image'].save(out_file)
                    results[img_file.name] = 'success'
                else:
                    results[img_file.name] = 'failed'
            except Exception as e:
                results[img_file.name] = f'error: {e}'
        return results 