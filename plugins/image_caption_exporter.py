from .plugin_base import PluginBase
from typing import Any, Dict
from pathlib import Path
from PIL import Image
import csv

class ImageCaptionExporterPlugin(PluginBase):
    """
    Plugin to caption all images in a folder using a vision-language agent and export results to CSV.
    """
    def run(self, input_dir: str, agent, output_csv: str = "captions.csv", **kwargs) -> Dict[str, Any]:
        input_path = Path(input_dir)
        results = []
        for img_file in input_path.glob("*.jpg"):
            try:
                image = Image.open(img_file)
                result = agent.process({'image': image, 'prompt': 'Describe this image', **kwargs})
                caption = result['result'] if isinstance(result, dict) else str(result)
                results.append({'file': img_file.name, 'caption': caption})
            except Exception as e:
                results.append({'file': img_file.name, 'caption': f'error: {e}'})
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'caption'])
            writer.writeheader()
            writer.writerows(results)
        return {'output_csv': output_csv, 'count': len(results)} 