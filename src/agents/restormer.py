from .base_agent import BaseAgent
import numpy as np
import torch
from typing import Dict, Any
import os

class RestormerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Restormer")

    async def _initialize(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model().to(self.device)
        self.model.eval()

    def load_model(self):
        from temp.Restormer.basicsr.models.archs.restormer_arch import Restormer
        model = Restormer()
        path_weights = 'weights/restormer-real-rain-streaks.pth'
        if os.path.exists(path_weights):
            model.load_state_dict(torch.load(path_weights)['params'])
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

    async def _cleanup(self) -> None:
        del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None