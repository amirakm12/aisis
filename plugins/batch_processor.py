from .plugin_base import PluginBase
from typing import Any, Dict
from pathlib import Path
from PIL import Image
import ray
from concurrent.futures import ThreadPoolExecutor

class BatchProcessorPlugin(PluginBase):
    """
    Plugin for batch processing images in a folder using a specified agent.
    """
    def run(self, input_dir: str, agent, output_dir: str = "output", num_workers: int = 4, use_ray: bool = False, **kwargs) -> Dict[str, Any]:
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        img_files = list(input_path.glob("*.jpg"))
        results = {}
        if use_ray:
            ray.init(ignore_reinit_error=True)
            @ray.remote
            def process_image(file, agent, kwargs, output_path):
                try:
                    image = Image.open(file)
                    result = agent.process({'image': image, **kwargs})
                    out_file = output_path / file.name
                    if isinstance(result, dict) and 'output_image' in result:
                        result['output_image'].save(out_file)
                        return file.name, 'success'
                    else:
                        return file.name, 'failed'
                except Exception as e:
                    return file.name, f'error: {e}'
            futures = [process_image.remote(file, agent, kwargs, output_path) for file in img_files]
            ray_results = ray.get(futures)
            results = dict(ray_results)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                def process(file):
                    try:
                        image = Image.open(file)
                        result = agent.process({'image': image, **kwargs})
                        out_file = output_path / file.name
                        if isinstance(result, dict) and 'output_image' in result:
                            result['output_image'].save(out_file)
                            return file.name, 'success'
                        else:
                            return file.name, 'failed'
                    except Exception as e:
                        return file.name, f'error: {e}'
                futures = [executor.submit(process, file) for file in img_files]
                for future in futures:
                    name, status = future.result()
                    results[name] = status
        return results 