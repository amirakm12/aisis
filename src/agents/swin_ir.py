from .base_agent import BaseAgent
import numpy as np
import torch
from typing import Dict, Any
import os

class SwinIRAgent(BaseAgent):
    def __init__(self):
        super().__init__("SwinIR")

    async def _initialize(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model().to(self.device)
        self.model.eval()

    def load_model(self):
        from temp.SwinIR.models.network_swinir import SwinIR
        model = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1., depths=[6,6,6,6,6,6,6,6,6], embed_dim=96, num_heads=[6,6,6,6,6,6,6,6,6], mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        path_weights = 'weights/swinir-realsr.pth'
        if os.path.exists(path_weights):
            model.load_state_dict(torch.load(path_weights))
        return model

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        input_image = task.get('image')
        if input_image is None:
            raise ValueError("No input image provided")
        input_tensor = torch.from_numpy(input_image).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
        with torch.no_grad():
            restored_tensor = self.model(input_tensor)
        restored_image = (restored_tensor.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        return {'restored_image': restored_image}

    async def _cleanup(self) -> None:
        del self.model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None