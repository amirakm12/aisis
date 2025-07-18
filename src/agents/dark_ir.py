from .base_agent import BaseAgent
import numpy as np
import torch
from typing import Dict, Any
import os

# Integrated from https://github.com/cidautai/DarkIR cloned to temp/DarkIR
# Note: Ensure dependencies are installed: pip install -r temp/DarkIR/requirements.txt

class DarkIRAgent(BaseAgent):
    def __init__(self):
        super().__init__("DarkIR")

    async def _initialize(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model().to(self.device)
        self.model.eval()

    def load_model(self):
        from temp.DarkIR.archs.retinexformer import RetinexFormer
        model = RetinexFormer()
        # Load weights - assume weights are downloaded to a path, e.g., 'weights/darkir.pth'
        path_weights = 'weights/darkir.pth'  # TODO: Download or provide path
        if os.path.exists(path_weights):
            checkpoints = torch.load(path_weights, map_location='cpu', weights_only=False)
            weights = checkpoints['params']
            weights = {'module.' + key: value for key, value in weights.items()}
            model.load_state_dict(weights)
        return model

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        input_image = task.get('image')
        if input_image is None:
            raise ValueError("No input image provided")
        input_tensor = self.preprocess(input_image).to(self.device)
        with torch.no_grad():
            restored_tensor = self.model(input_tensor)
        restored_image = self.postprocess(restored_tensor)
        return {'output_image': restored_image, 'status': 'completed'}

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0

    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        return (tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    async def _cleanup(self) -> None:
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()