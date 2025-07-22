"""
Context-Aware Restoration Agent
Specialized agent for intelligent restoration based on image context and content
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import cv2
from loguru import logger

from .base_agent import BaseAgent
from ..core.gpu_utils import gpu_manager


class ContextAwareRestorationAgent(BaseAgent):
    def __init__(self):
        super().__init__("ContextAwareRestorationAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.context_types = [
            "portrait",
            "landscape",
            "document",
            "artwork",
            "photograph",
            "text",
            "object",
            "scene",
            "texture",
            "pattern",
        ]

    async def _initialize(self) -> None:
        """Initialize context-aware restoration models"""
        try:
            # Context classification network
            self.models["context_classifier"] = await self._load_context_classifier()

            # Content-aware restoration network
            self.models["content_restorer"] = await self._load_content_restorer()

            # Semantic segmentation network
            self.models["semantic_segmenter"] = await self._load_semantic_segmenter()

            # Adaptive restoration network
            self.models["adaptive_restorer"] = await self._load_adaptive_restorer()

            # Context-specific enhancement network
            self.models["context_enhancer"] = await self._load_context_enhancer()

            logger.info("Context-aware restoration models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize context-aware restoration models: {e}")
            raise

    async def _load_context_classifier(self) -> nn.Module:
        """Load context classification network"""

        class ContextClassifier(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )

                self.classifier = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, num_classes),
                    nn.Softmax(dim=1),
                )

            def forward(self, x):
                features = self.features(x)
                features = features.view(features.size(0), -1)
                context_probs = self.classifier(features)
                return context_probs

        return ContextClassifier(len(self.context_types)).to(self.device)

    async def _load_content_restorer(self) -> nn.Module:
        """Load content-aware restoration network"""

        class ContentRestorer(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, 3, padding=1),
                    nn.Tanh(),
                )

            def forward(self, x):
                encoded = self.encoder(x)
                restored = self.decoder(encoded)
                return restored

        return ContentRestorer().to(self.device)

    async def _load_semantic_segmenter(self) -> nn.Module:
        """Load semantic segmentation network"""

        class SemanticSegmenter(nn.Module):
            def __init__(self, num_classes=21):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, num_classes, 1),
                    nn.Softmax(dim=1),
                )

            def forward(self, x):
                encoded = self.encoder(x)
                segmentation = self.decoder(encoded)
                return segmentation

        return SemanticSegmenter().to(self.device)

    async def _load_adaptive_restorer(self) -> nn.Module:
        """Load adaptive restoration network"""

        class AdaptiveRestorer(nn.Module):
            def __init__(self):
                super().__init__()
                self.context_encoder = nn.Sequential(
                    nn.Linear(10, 64),  # 10 context types
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 128),
                    nn.ReLU(inplace=True),
                )

                self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

                self.fusion = nn.Sequential(
                    nn.Conv2d(256 + 128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                )

                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, 3, padding=1),
                    nn.Tanh(),
                )

            def forward(self, x, context):
                image_features = self.image_encoder(x)
                context_features = self.context_encoder(context)

                # Reshape context features to match spatial dimensions
                batch_size, channels, height, width = image_features.shape
                context_features = context_features.unsqueeze(-1).unsqueeze(-1)
                context_features = context_features.expand(-1, -1, height, width)

                # Fuse features
                fused = torch.cat([image_features, context_features], dim=1)
                fused = self.fusion(fused)

                # Decode
                restored = self.decoder(fused)
                return restored

        return AdaptiveRestorer().to(self.device)

    async def _load_context_enhancer(self) -> nn.Module:
        """Load context-specific enhancement network"""

        class ContextEnhancer(nn.Module):
            def __init__(self):
                super().__init__()
                self.enhancer = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 3, 3, padding=1),
                    nn.Tanh(),
                )

            def forward(self, x):
                enhanced = self.enhancer(x)
                return enhanced

        return ContextEnhancer().to(self.device)

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process context-aware restoration task"""
        try:
            # Get input image
            image = task.get("image")
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Convert to tensor
            x = self._image_to_tensor(image).unsqueeze(0).to(self.device)

            # Classify context
            context_probs = await self._classify_context(x)
            context_type = self._get_primary_context(context_probs)

            # Perform semantic segmentation
            segmentation = await self._segment_semantics(x)

            # Apply content-aware restoration
            content_restored = await self._restore_content(x)

            # Apply adaptive restoration based on context
            adaptive_restored = await self._apply_adaptive_restoration(x, context_probs)

            # Apply context-specific enhancement
            enhanced = await self._apply_context_enhancement(adaptive_restored)

            # Convert back to image
            output_image = self._tensor_to_image(enhanced)

            # Generate context analysis
            context_analysis = self._analyze_context(context_probs, segmentation)

            # Save if output path provided
            output_path = None
            if "output_path" in task:
                output_path = task["output_path"]
                output_image.save(output_path)

            return {
                "status": "success",
                "output_image": output_image,
                "output_path": output_path,
                "context_type": context_type,
                "context_probabilities": context_probs.tolist(),
                "context_analysis": context_analysis,
                "semantic_segmentation": (
                    segmentation.tolist() if segmentation is not None else None
                ),
                "restoration_methods": ["content_aware", "adaptive", "context_enhancement"],
            }

        except Exception as e:
            logger.error(f"Context-aware restoration failed: {e}")
            raise

    async def _classify_context(self, x: torch.Tensor) -> torch.Tensor:
        """Classify image context"""
        with torch.no_grad():
            return self.models["context_classifier"](x)

    async def _segment_semantics(self, x: torch.Tensor) -> torch.Tensor:
        """Perform semantic segmentation"""
        with torch.no_grad():
            return self.models["semantic_segmenter"](x)

    async def _restore_content(self, x: torch.Tensor) -> torch.Tensor:
        """Apply content-aware restoration"""
        with torch.no_grad():
            return self.models["content_restorer"](x)

    async def _apply_adaptive_restoration(
        self, x: torch.Tensor, context_probs: torch.Tensor
    ) -> torch.Tensor:
        """Apply adaptive restoration based on context"""
        with torch.no_grad():
            return self.models["adaptive_restorer"](x, context_probs)

    async def _apply_context_enhancement(self, x: torch.Tensor) -> torch.Tensor:
        """Apply context-specific enhancement"""
        with torch.no_grad():
            return self.models["context_enhancer"](x)

    def _get_primary_context(self, context_probs: torch.Tensor) -> str:
        """Get primary context type"""
        probs = context_probs.squeeze().cpu().numpy()
        primary_idx = np.argmax(probs)
        return self.context_types[primary_idx]

    def _analyze_context(
        self, context_probs: torch.Tensor, segmentation: torch.Tensor
    ) -> Dict[str, Any]:
        """Analyze image context"""
        probs = context_probs.squeeze().cpu().numpy()

        analysis = {
            "primary_context": self._get_primary_context(context_probs),
            "context_confidence": float(np.max(probs)),
            "context_distribution": dict(zip(self.context_types, probs.tolist())),
            "secondary_contexts": [],
        }

        # Get secondary contexts
        sorted_indices = np.argsort(probs)[::-1]
        for i in range(1, min(4, len(sorted_indices))):
            if probs[sorted_indices[i]] > 0.1:
                analysis["secondary_contexts"].append(
                    {
                        "type": self.context_types[sorted_indices[i]],
                        "confidence": float(probs[sorted_indices[i]]),
                    }
                )

        return analysis

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor"""
        # Resize to standard size
        image = image.resize((512, 512))
        # Convert to tensor
        x = torch.from_numpy(np.array(image)).float() / 255.0
        x = x.permute(2, 0, 1)  # HWC to CHW
        return x

    def _tensor_to_image(self, x: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        x = x.squeeze(0).cpu()
        x = torch.clamp(x, -1, 1)  # Tanh output
        x = (x + 1) / 2  # Convert to [0, 1]
        x = (x * 255).byte()
        x = x.permute(1, 2, 0)  # CHW to HWC
        return Image.fromarray(x.numpy())

    async def _cleanup(self) -> None:
        """Cleanup models and resources"""
        self.models.clear()
        torch.cuda.empty_cache()
