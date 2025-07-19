"""
Style and Aesthetic Agent
Autonomously improves images based on aesthetic scoring and style transfer
"""

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Dict, Any
from loguru import logger

from .base_agent import BaseAgent
from ..core.gpu_utils import gpu_manager


class StyleAestheticAgent(BaseAgent):
    def __init__(self):
        super().__init__("StyleAestheticAgent")
        self.device = gpu_manager.device
        self.model = None
        self.transforms = None

    async def _initialize(self) -> None:
        """Initialize style and aesthetic models"""
        try:
            # Placeholder for loading style transfer and aesthetic scoring models
            self.model = await self._load_model()
            self.transforms = T.Compose(
                [
                    T.Resize((512, 512)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            logger.info("Style and Aesthetic models initialized")
        except Exception as e:
            logger.error(f"Failed to initialize StyleAestheticAgent: {e}")
            raise

    async def _load_model(self):
        # Dummy model placeholder
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return x

        model = DummyModel().to(self.device)
        return model

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process style and aesthetic enhancement task"""
        try:
            image = task.get("image")
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            x = self.transforms(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(x)

            output_image = self._tensor_to_image(output)

            output_path = None
            if "output_path" in task:
                output_path = task["output_path"]
                output_image.save(output_path)

            return {
                "status": "success",
                "output_image": output_image,
                "output_path": output_path,
            }
        except Exception as e:
            logger.error(f"Style and Aesthetic processing failed: {e}")
            raise

    def _tensor_to_image(self, x: torch.Tensor) -> Image.Image:
        x = x.squeeze(0).cpu()
        x = torch.clamp(x, 0, 1)
        x = (x * 255).byte()
        return Image.fromarray(x.permute(1, 2, 0).numpy())

    async def _cleanup(self) -> None:
        self.model = None
        torch.cuda.empty_cache()
