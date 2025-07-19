"""
Perspective Correction Agent
Detects and corrects skewed or warped elements
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

from .base_agent import BaseAgent
from ..core.gpu_utils import gpu_manager


class PerspectiveCorrectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("PerspectiveCorrectionAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.transforms = None

    async def _initialize(self) -> None:
        """Initialize perspective correction models"""
        try:
            # TODO: Replace with real perspective correction models
            logger.warning(
                "Perspective correction models are placeholders. Implement real models."
            )

            # Perspective detection model
            self.models["perspective_detector"] = (
                await self._load_perspective_detector()
            )

            # Corner detection model
            self.models["corner_detector"] = await self._load_corner_detector()

            # Homography estimation model
            self.models["homography_estimator"] = (
                await self._load_homography_estimator()
            )

            # Grid detection model
            self.models["grid_detector"] = await self._load_grid_detector()

            # Setup transforms
            self.transforms = T.Compose(
                [
                    T.Resize((512, 512)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            logger.info("Perspective correction models initialized")

        except Exception as e:
            logger.error(
                f"Failed to initialize perspective correction models: {e}"
            )
            raise

    async def _load_perspective_detector(self) -> nn.Module:
        """Load perspective distortion detection model"""

        class PerspectiveDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 4)  # 4 perspective parameters

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).squeeze(-1).squeeze(-1)
                return torch.tanh(self.fc(x))  # Normalized output

        return PerspectiveDetector().to(self.device)

    async def _load_corner_detector(self) -> nn.Module:
        """Load corner detection model"""

        class CornerDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 1, 1)  # Corner heatmap

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = torch.relu(self.conv2(x))
                return torch.sigmoid(self.conv3(x))

        return CornerDetector().to(self.device)

    async def _load_homography_estimator(self) -> nn.Module:
        """Load homography estimation model"""

        class HomographyEstimator(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(6, 64, 3, padding=1)  # 2 images
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 8)  # 8 homography parameters

            def forward(self, x1, x2):
                x = torch.cat([x1, x2], dim=1)
                x = torch.relu(self.conv(x))
                x = self.pool(x).squeeze(-1).squeeze(-1)
                return self.fc(x)

        return HomographyEstimator().to(self.device)

    async def _load_grid_detector(self) -> nn.Module:
        """Load grid detection model"""

        class GridDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
                self.conv3 = nn.Conv2d(32, 2, 1)  # Grid lines

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = torch.relu(self.conv2(x))
                return torch.sigmoid(self.conv3(x))

        return GridDetector().to(self.device)

    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process perspective correction task"""
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
            task_type = task.get("task_type", "auto_correct")

            if task_type == "detect_perspective":
                result = await self._detect_perspective(x)
            elif task_type == "detect_corners":
                result = await self._detect_corners(x)
            elif task_type == "detect_grid":
                result = await self._detect_grid(x)
            elif task_type == "manual_correct":
                corners = task.get("corners")
                result = await self._manual_perspective_correct(image, corners)
            else:
                # Auto correction
                result = await self._auto_perspective_correct(x, image)

            # Convert back to image if needed
            if "corrected_image" in result:
                output_image = result["corrected_image"]
                result["output_image"] = output_image

                # Save if output path provided
                if "output_path" in task:
                    output_path = task["output_path"]
                    output_image.save(output_path)
                    result["output_path"] = output_path

            return result

        except Exception as e:
            logger.error(f"Perspective correction failed: {e}")
            raise

    async def _detect_perspective(self, x: torch.Tensor) -> Dict[str, Any]:
        """Detect perspective distortion"""
        with torch.no_grad():
            perspective_params = self.models["perspective_detector"](x)

        return {
            "status": "success",
            "perspective_parameters": perspective_params.cpu().numpy(),
            "distortion_level": float(torch.norm(perspective_params).item()),
        }

    async def _detect_corners(self, x: torch.Tensor) -> Dict[str, Any]:
        """Detect corners in image"""
        with torch.no_grad():
            corner_heatmap = self.models["corner_detector"](x)

        # Convert heatmap to corner coordinates
        corners = self._heatmap_to_corners(corner_heatmap)

        return {
            "status": "success",
            "corners": corners,
            "corner_heatmap": corner_heatmap.cpu().numpy(),
        }

    async def _detect_grid(self, x: torch.Tensor) -> Dict[str, Any]:
        """Detect grid lines in image"""
        with torch.no_grad():
            grid_lines = self.models["grid_detector"](x)

        return {"status": "success", "grid_lines": grid_lines.cpu().numpy()}

    async def _auto_perspective_correct(
        self, x: torch.Tensor, original_image: Image.Image
    ) -> Dict[str, Any]:
        """Automatically correct perspective"""
        # Detect perspective
        perspective_result = await self._detect_perspective(x)

        # Detect corners
        corner_result = await self._detect_corners(x)

        # If we have enough corners, use them for correction
        if len(corner_result["corners"]) >= 4:
            corrected_image = await self._correct_with_corners(
                original_image, corner_result["corners"]
            )
        else:
            # Use perspective parameters for correction
            corrected_image = await self._correct_with_parameters(
                original_image, perspective_result["perspective_parameters"]
            )

        return {
            "status": "success",
            "corrected_image": corrected_image,
            "perspective_parameters": perspective_result[
                "perspective_parameters"
            ],
            "detected_corners": corner_result["corners"],
        }

    async def _manual_perspective_correct(
        self, image: Image.Image, corners: List[Tuple[int, int]]
    ) -> Dict[str, Any]:
        """Manually correct perspective using provided corners"""
        if len(corners) != 4:
            raise ValueError(
                "Exactly 4 corners required for perspective correction"
            )

        corrected_image = await self._correct_with_corners(image, corners)

        return {
            "status": "success",
            "corrected_image": corrected_image,
            "used_corners": corners,
        }

    async def _correct_with_corners(
        self, image: Image.Image, corners: List[Tuple[int, int]]
    ) -> Image.Image:
        """Correct perspective using corner points"""
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Define target corners (rectangle)
        h, w = img_cv.shape[:2]
        target_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        source_corners = np.float32(corners)

        # Calculate homography
        homography = cv2.getPerspectiveTransform(
            source_corners, target_corners
        )

        # Apply perspective correction
        corrected_cv = cv2.warpPerspective(img_cv, homography, (w, h))

        # Convert back to PIL
        corrected_image = Image.fromarray(
            cv2.cvtColor(corrected_cv, cv2.COLOR_BGR2RGB)
        )

        return corrected_image

    async def _correct_with_parameters(
        self, image: Image.Image, params: np.ndarray
    ) -> Image.Image:
        """Correct perspective using estimated parameters"""
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create transformation matrix from parameters
        # This is a simplified approach - real implementation would be more sophisticated
        h, w = img_cv.shape[:2]

        # Apply simple affine transformation based on parameters
        matrix = np.array(
            [
                [1 + params[0], params[1], params[2]],
                [params[3], 1 + params[0], params[2]],
            ],
            dtype=np.float32,
        )

        corrected_cv = cv2.warpAffine(img_cv, matrix, (w, h))

        # Convert back to PIL
        corrected_image = Image.fromarray(
            cv2.cvtColor(corrected_cv, cv2.COLOR_BGR2RGB)
        )

        return corrected_image

    def _heatmap_to_corners(
        self, heatmap: torch.Tensor, threshold: float = 0.5
    ) -> List[Tuple[int, int]]:
        """Convert corner heatmap to corner coordinates"""
        heatmap_np = heatmap.squeeze().cpu().numpy()

        # Find local maxima
        corners = []
        h, w = heatmap_np.shape

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if heatmap_np[i, j] > threshold:
                    # Check if it's a local maximum
                    if (
                        heatmap_np[i, j]
                        >= heatmap_np[i - 1 : i + 2, j - 1 : j + 2]
                    ).all():
                        # Convert to original image coordinates
                        x = int(j * w / heatmap_np.shape[1])
                        y = int(i * h / heatmap_np.shape[0])
                        corners.append((x, y))

        # Sort corners by confidence and take top 4
        corners = sorted(
            corners, key=lambda c: heatmap_np[c[1], c[0]], reverse=True
        )[:4]

        return corners

    async def _cleanup(self) -> None:
        """Cleanup models and resources"""
        self.models.clear()
        torch.cuda.empty_cache()
