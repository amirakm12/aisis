"""
Self-Critique Agent
Specialized agent for continuous quality assessment and improvement feedback
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


class SelfCritiqueAgent(BaseAgent):
    def __init__(self):
        super().__init__("SelfCritiqueAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.critique_dimensions = [
            "sharpness",
            "noise_level",
            "color_accuracy",
            "contrast_balance",
            "detail_preservation",
            "artifacts",
            "consistency",
            "aesthetic_quality",
            "technical_accuracy",
            "restoration_fidelity",
        ]

    async def _initialize(self) -> None:
        """Initialize self-critique models"""
        try:
            # Multi-dimensional critique network
            self.models["critique_network"] = (
                await self._load_critique_network()
            )

            # Quality improvement predictor
            self.models["improvement_predictor"] = (
                await self._load_improvement_predictor()
            )

            # Feedback generator
            self.models["feedback_generator"] = (
                await self._load_feedback_generator()
            )

            # Iterative improvement network
            self.models["iterative_improver"] = (
                await self._load_iterative_improver()
            )

            logger.info("Self-critique models initialized")

        except Exception as e:
            logger.error(f"Failed to initialize self-critique models: {e}")
            raise

    async def _load_critique_network(self) -> nn.Module:
        """Load multi-dimensional critique network"""

        class CritiqueNetwork(nn.Module):
            def __init__(self, num_dimensions=10):
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

                self.critique_heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(512, 256),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(256, 128),
                            nn.ReLU(inplace=True),
                            nn.Linear(128, 1),
                            nn.Sigmoid(),
                        )
                        for _ in range(num_dimensions)
                    ]
                )

            def forward(self, x):
                features = self.features(x)
                features = features.view(features.size(0), -1)

                critiques = []
                for head in self.critique_heads:
                    critique = head(features)
                    critiques.append(critique)

                return torch.cat(critiques, dim=1)

        return CritiqueNetwork(len(self.critique_dimensions)).to(self.device)

    async def _load_improvement_predictor(self) -> nn.Module:
        """Load improvement prediction network"""

        class ImprovementPredictor(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
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
                    nn.AdaptiveAvgPool2d((1, 1)),
                )

                self.predictor = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 10),  # 10 improvement scores
                    nn.Sigmoid(),
                )

            def forward(self, x):
                features = self.features(x)
                features = features.view(features.size(0), -1)
                return self.predictor(features)

        return ImprovementPredictor().to(self.device)

    async def _load_feedback_generator(self) -> nn.Module:
        """Load feedback generation network"""

        class FeedbackGenerator(nn.Module):
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
                )

                self.feedback_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32),  # Feedback embedding
                    nn.Tanh(),
                )

            def forward(self, x):
                features = self.encoder(x)
                feedback = self.feedback_head(features)
                return feedback

        return FeedbackGenerator().to(self.device)

    async def _load_iterative_improver(self) -> nn.Module:
        """Load iterative improvement network"""

        class IterativeImprover(nn.Module):
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

            def forward(self, x):
                encoded = self.encoder(x)
                improved = self.decoder(encoded)
                return improved

        return IterativeImprover().to(self.device)

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process self-critique task"""
        try:
            # Get input image
            image = task.get("image")
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Convert to tensor
            x = self._image_to_tensor(image).unsqueeze(0).to(self.device)

            # Perform multi-dimensional critique
            critique_scores = await self._perform_critique(x)

            # Generate improvement predictions
            improvement_scores = await self._predict_improvements(x)

            # Generate feedback embedding
            feedback_embedding = await self._generate_feedback(x)

            # Determine if iterative improvement is needed
            needs_improvement = torch.any(improvement_scores > 0.3)

            improved_image = None
            improvement_applied = False

            if needs_improvement:
                # Apply iterative improvement
                improved_image = await self._apply_iterative_improvement(x)
                improvement_applied = True

                # Re-assess after improvement
                improved_critique = await self._perform_critique(
                    improved_image
                )

                # Convert back to image
                improved_image = self._tensor_to_image(improved_image)
            else:
                improved_image = image
                improved_critique = critique_scores

            # Create detailed feedback report
            feedback_report = self._create_feedback_report(
                critique_scores, improvement_scores, self.critique_dimensions
            )

            # Save if output path provided
            output_path = None
            if "output_path" in task:
                output_path = task["output_path"]
                improved_image.save(output_path)

            return {
                "status": "success",
                "output_image": improved_image,
                "output_path": output_path,
                "critique_scores": critique_scores.tolist(),
                "improvement_scores": improvement_scores.tolist(),
                "feedback_embedding": feedback_embedding.tolist(),
                "needs_improvement": bool(needs_improvement),
                "improvement_applied": improvement_applied,
                "improved_critique": (
                    improved_critique.tolist()
                    if improved_critique is not None
                    else None
                ),
                "feedback_report": feedback_report,
            }

        except Exception as e:
            logger.error(f"Self-critique failed: {e}")
            raise

    async def _perform_critique(self, x: torch.Tensor) -> torch.Tensor:
        """Perform multi-dimensional critique"""
        with torch.no_grad():
            return self.models["critique_network"](x)

    async def _predict_improvements(self, x: torch.Tensor) -> torch.Tensor:
        """Predict improvement potential"""
        with torch.no_grad():
            return self.models["improvement_predictor"](x)

    async def _generate_feedback(self, x: torch.Tensor) -> torch.Tensor:
        """Generate feedback embedding"""
        with torch.no_grad():
            return self.models["feedback_generator"](x)

    async def _apply_iterative_improvement(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Apply iterative improvement"""
        with torch.no_grad():
            return self.models["iterative_improver"](x)

    def _create_feedback_report(
        self,
        critique_scores: torch.Tensor,
        improvement_scores: torch.Tensor,
        dimensions: List[str],
    ) -> Dict[str, Any]:
        """Create detailed feedback report"""
        report = {
            "overall_quality": float(torch.mean(critique_scores)),
            "dimension_analysis": {},
            "recommendations": [],
        }

        for i, dimension in enumerate(dimensions):
            critique_score = float(critique_scores[0, i])
            improvement_score = float(improvement_scores[0, i])

            report["dimension_analysis"][dimension] = {
                "current_score": critique_score,
                "improvement_potential": improvement_score,
                "status": self._get_status(critique_score, improvement_score),
            }

            if improvement_score > 0.3:
                report["recommendations"].append(
                    f"Improve {dimension}: Current score {critique_score:.2f}, "
                    f"improvement potential {improvement_score:.2f}"
                )

        return report

    def _get_status(
        self, critique_score: float, improvement_score: float
    ) -> str:
        """Get status based on scores"""
        if critique_score > 0.8:
            return "excellent"
        elif critique_score > 0.6:
            return "good"
        elif critique_score > 0.4:
            return "fair"
        else:
            return "needs_improvement"

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
