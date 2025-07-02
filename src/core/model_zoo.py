from typing import Dict, List


class ModelZoo:
    """
    Model Zoo for managing available models (vision, LLM, ASR, etc.).
    Supports browsing, downloading, updating, and switching models.
    """

    def __init__(self):
        self.registry: Dict[str, Dict] = {}

    def register_model(self, name: str, info: Dict):
        self.registry[name] = info

    def list_models(self, model_type: str = None) -> List[str]:
        if model_type:
            return [k for k, v in self.registry.items() if v.get('type') == model_type]
        return list(self.registry.keys())

    def get_model_info(self, name: str) -> Dict:
        return self.registry.get(name, {})

    def download_model(self, name: str):
        # TODO: Download model from remote or local source
        pass

    def switch_model(self, name: str):
        # TODO: Switch active model for a given task/agent
        pass 