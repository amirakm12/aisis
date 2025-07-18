from .base_agent import BaseAgent
import numpy as np
import torch
from typing import Dict, Any
import os

# Integrated from https://github.com/shallowdream204/DreamClear cloned to temp/DreamClear
# Note: Ensure dependencies are installed: pip install -r temp/DreamClear/requirements.txt

class DreamClearAgent(BaseAgent):
    def __init__(self):
        super().__init__("DreamClear")

    async def _initialize(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model().to(self.device)
        self.model.eval()

    def load_model(self):
        from temp.DreamClear.models.dreamclear import DreamClear
        model = DreamClear()
        path_weights = 'weights/dreamclear-1024.pth'
        if os.path.exists(path_weights):
            model.load_state_dict(torch.load(path_weights))
        return model

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        input_image = task.get('image')
        if input_image is None:
            raise ValueError("No input image provided")
        input_tensor = torch.from_numpy(input_image).float().permute(2,0,1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            restored_tensor = self.model(input_tensor)
        restored_image = restored_tensor.squeeze(0).permute(1,2,0).cpu().numpy()
        return {'restored_image': restored_image}

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        return (tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    async def _cleanup(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()