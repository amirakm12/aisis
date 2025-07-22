"""
Material Recognition Agent
Classifies regions by material type (metal, paper, cloth, etc.) and applies restoration filters intelligently
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple
from loguru import logger

from .base_agent import BaseAgent
from ..core.gpu_utils import gpu_manager


class MaterialRecognitionAgent(BaseAgent):
    def __init__(self):
        super().__init__("MaterialRecognitionAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.transforms = None
        self.material_classes = [
            "metal",
            "paper",
            "cloth",
            "wood",
            "stone",
            "ceramic",
            "glass",
            "plastic",
            "leather",
            "paint",
            "ink",
            "fabric",
            "parchment",
            "canvas",
            "marble",
            "bronze",
            "silver",
            "gold",
        ]

    async def _initialize(self) -> None:
        """Initialize material recognition models"""
        try:
            # TODO: Replace with real material recognition models
            logger.warning("Material recognition models are placeholders. Implement real models.")

            # Material classifier
            self.models["material_classifier"] = await self._load_material_classifier()

            # Texture analyzer
            self.models["texture_analyzer"] = await self._load_texture_analyzer()

            # Surface property detector
            self.models["surface_properties"] = await self._load_surface_properties()

            # Material-specific restoration
            self.models["material_restoration"] = await self._load_material_restoration()

            # Setup transforms
            self.transforms = T.Compose(
                [
                    T.Resize((512, 512)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            logger.info("Material recognition models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize material recognition models: {e}")
            raise

    async def _load_material_classifier(self) -> nn.Module:
        """Load material classification model"""

        class MaterialClassifier(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.classifier = nn.Linear(128, num_classes)

            def forward(self, x):
                features = self.backbone(x)
                features = features.squeeze(-1).squeeze(-1)
                return torch.softmax(self.classifier(features), dim=1)

        return MaterialClassifier(len(self.material_classes)).to(self.device)

    async def _load_texture_analyzer(self) -> nn.Module:
        """Load texture analysis model"""

        class TextureAnalyzer(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 10)  # Texture properties

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).squeeze(-1).squeeze(-1)
                return self.fc(x)

        return TextureAnalyzer().to(self.device)

    async def _load_surface_properties(self) -> nn.Module:
        """Load surface property detection model"""

        class SurfaceProperties(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 32, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(32, 8)  # Properties: roughness, reflectivity, etc.

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).squeeze(-1).squeeze(-1)
                return torch.sigmoid(self.fc(x))

        return SurfaceProperties().to(self.device)

    async def _load_material_restoration(self) -> nn.Module:
        """Load material-specific restoration model"""

        class MaterialRestoration(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(6, 64, 3, padding=1)  # image + material mask
                self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 3, 3, padding=1)

            def forward(self, x, material_mask):
                x = torch.cat([x, material_mask], dim=1)
                x = torch.relu(self.conv(x))
                x = torch.relu(self.conv2(x))
                return self.conv3(x)

        return MaterialRestoration().to(self.device)

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process material recognition task"""
        try:
            # Get input image
            image = task.get("image")
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Convert to tensor
            x = self.transforms(image).unsqueeze(0).to(self.device)

            # Get task type
            task_type = task.get("task_type", "classify_and_restore")

            if task_type == "classify":
                result = await self._classify_materials(x)
            elif task_type == "analyze_texture":
                result = await self._analyze_texture(x)
            elif task_type == "detect_properties":
                result = await self._detect_surface_properties(x)
            else:
                # Full pipeline
                result = await self._full_material_analysis(x, image)

            # Convert back to image if needed
            if "restored_image" in result:
                output_image = result["restored_image"]
                result["output_image"] = output_image

                # Save if output path provided
                if "output_path" in task:
                    output_path = task["output_path"]
                    output_image.save(output_path)
                    result["output_path"] = output_path

            return result

        except Exception as e:
            logger.error(f"Material recognition failed: {e}")
            raise

    async def _classify_materials(self, x: torch.Tensor) -> Dict[str, Any]:
        """Classify materials in image"""
        with torch.no_grad():
            material_probs = self.models["material_classifier"](x)

        # Get top materials
        top_probs, top_indices = torch.topk(material_probs, 5, dim=1)

        materials = []
        for i in range(top_indices.shape[1]):
            material_name = self.material_classes[top_indices[0, i].item()]
            confidence = top_probs[0, i].item()
            materials.append({"material": material_name, "confidence": confidence})

        return {
            "status": "success",
            "materials": materials,
            "material_probabilities": material_probs.cpu().numpy(),
        }

    async def _analyze_texture(self, x: torch.Tensor) -> Dict[str, Any]:
        """Analyze texture properties"""
        with torch.no_grad():
            texture_features = self.models["texture_analyzer"](x)

        texture_properties = {
            "roughness": texture_features[0, 0].item(),
            "smoothness": texture_features[0, 1].item(),
            "granularity": texture_features[0, 2].item(),
            "directionality": texture_features[0, 3].item(),
            "complexity": texture_features[0, 4].item(),
        }

        return {
            "status": "success",
            "texture_properties": texture_properties,
            "texture_features": texture_features.cpu().numpy(),
        }

    async def _detect_surface_properties(self, x: torch.Tensor) -> Dict[str, Any]:
        """Detect surface properties"""
        with torch.no_grad():
            properties = self.models["surface_properties"](x)

        surface_properties = {
            "roughness": properties[0, 0].item(),
            "reflectivity": properties[0, 1].item(),
            "porosity": properties[0, 2].item(),
            "hardness": properties[0, 3].item(),
            "absorption": properties[0, 4].item(),
            "transparency": properties[0, 5].item(),
            "metallic": properties[0, 6].item(),
            "organic": properties[0, 7].item(),
        }

        return {
            "status": "success",
            "surface_properties": surface_properties,
            "property_vector": properties.cpu().numpy(),
        }

    async def _full_material_analysis(
        self, x: torch.Tensor, original_image: Image.Image
    ) -> Dict[str, Any]:
        """Full material analysis and restoration pipeline"""
        # Classify materials
        material_result = await self._classify_materials(x)

        # Analyze texture
        texture_result = await self._analyze_texture(x)

        # Detect surface properties
        properties_result = await self._detect_surface_properties(x)

        # Create material-specific restoration
        restored_image = await self._apply_material_restoration(x, material_result)

        return {
            "status": "success",
            "materials": material_result["materials"],
            "texture_properties": texture_result["texture_properties"],
            "surface_properties": properties_result["surface_properties"],
            "restored_image": restored_image,
            "recommended_treatments": self._get_recommended_treatments(
                material_result["materials"][0]["material"],
                texture_result["texture_properties"],
                properties_result["surface_properties"],
            ),
        }

    async def _apply_material_restoration(
        self, x: torch.Tensor, material_result: Dict[str, Any]
    ) -> Image.Image:
        """Apply material-specific restoration"""
        # Create material mask (simplified - in reality would be segmentation)
        material_mask = torch.zeros(1, 3, x.shape[2], x.shape[3]).to(self.device)

        with torch.no_grad():
            restored = self.models["material_restoration"](x, material_mask)

        return self._tensor_to_image(restored)

    def _get_recommended_treatments(
        self, material: str, texture_props: Dict[str, float], surface_props: Dict[str, float]
    ) -> List[str]:
        """Get recommended restoration treatments based on material analysis"""
        treatments = []

        if material == "paper":
            if surface_props["porosity"] > 0.7:
                treatments.append("gentle_cleaning")
                treatments.append("deacidification")
            if texture_props["roughness"] > 0.5:
                treatments.append("surface_smoothing")

        elif material == "metal":
            if surface_props["metallic"] > 0.8:
                treatments.append("corrosion_removal")
                treatments.append("protective_coating")
            if surface_props["reflectivity"] > 0.6:
                treatments.append("polishing")

        elif material == "cloth":
            if texture_props["complexity"] > 0.7:
                treatments.append("fiber_analysis")
                treatments.append("gentle_washing")
            if surface_props["absorption"] > 0.5:
                treatments.append("stain_removal")

        elif material == "paint":
            treatments.append("pigment_analysis")
            treatments.append("layer_separation")
            if surface_props["hardness"] < 0.3:
                treatments.append("consolidation")

        return treatments

    def _tensor_to_image(self, x: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        x = x.squeeze(0).cpu()
        x = torch.clamp(x, 0, 1)
        x = (x * 255).byte()
        return Image.fromarray(x.permute(1, 2, 0).numpy())

    async def _cleanup(self) -> None:
        """Cleanup models and resources"""
        self.models.clear()
        torch.cuda.empty_cache()
