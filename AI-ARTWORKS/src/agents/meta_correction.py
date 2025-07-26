"""
Meta-Correction Agent
Flagship agent for self-critique, meta-level corrections, and quality assurance
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

class MetaCorrectionAgent(BaseAgent):
    def __init__(self):
        super().__init__("MetaCorrectionAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.quality_threshold = 0.85
        
    async def _initialize(self) -> None:
        """Initialize meta-correction models"""
        try:
            # Quality assessment network
            self.models['quality_assessor'] = await self._load_quality_assessor()
            
            # Self-critique network
            self.models['self_critique'] = await self._load_self_critique()
            
            # Meta-correction network
            self.models['meta_correction'] = await self._load_meta_correction()
            
            # Consistency checker
            self.models['consistency_checker'] = await self._load_consistency_checker()
            
            # Error detection network
            self.models['error_detector'] = await self._load_error_detector()
            
            logger.info("Meta-correction models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize meta-correction models: {e}")
            raise
    
    async def _load_quality_assessor(self) -> nn.Module:
        """Load quality assessment network"""
        class QualityAssessor(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 7, stride=2, padding=3),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, stride=2, padding=1),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)
        
        return QualityAssessor().to(self.device)
    
    async def _load_self_critique(self) -> nn.Module:
        """Load self-critique network"""
        class SelfCritique(nn.Module):
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
                    nn.ReLU(inplace=True)
                )
                
                self.critique_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 10)  # 10 critique dimensions
                )
            
            def forward(self, x):
                features = self.encoder(x)
                critique = self.critique_head(features)
                return critique
        
        return SelfCritique().to(self.device)
    
    async def _load_meta_correction(self) -> nn.Module:
        """Load meta-correction network"""
        class MetaCorrection(nn.Module):
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
                    nn.ReLU(inplace=True)
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
                    nn.Tanh()
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                corrected = self.decoder(encoded)
                return corrected
        
        return MetaCorrection().to(self.device)
    
    async def _load_consistency_checker(self) -> nn.Module:
        """Load consistency checker network"""
        class ConsistencyChecker(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(6, 64, 3, padding=1),  # 6 channels for before/after
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, before, after):
                combined = torch.cat([before, after], dim=1)
                features = self.features(combined)
                consistency = self.classifier(features.view(features.size(0), -1))
                return consistency
        
        return ConsistencyChecker().to(self.device)
    
    async def _load_error_detector(self) -> nn.Module:
        """Load error detection network"""
        class ErrorDetector(nn.Module):
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
                    nn.ReLU(inplace=True)
                )
                
                self.error_head = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                features = self.features(x)
                error_map = self.error_head(features)
                return error_map
        
        return ErrorDetector().to(self.device)
    
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process meta-correction task"""
        try:
            # Get input image
            image = task.get('image')
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to tensor
            x = self._image_to_tensor(image).unsqueeze(0).to(self.device)
            
            # Perform quality assessment
            quality_score = await self._assess_quality(x)
            
            # Perform self-critique
            critique = await self._perform_self_critique(x)
            
            # Check if correction is needed
            needs_correction = quality_score < self.quality_threshold
            
            corrected_image = None
            correction_applied = False
            
            if needs_correction:
                # Apply meta-correction
                corrected_image = await self._apply_meta_correction(x)
                correction_applied = True
                
                # Check consistency
                consistency_score = await self._check_consistency(x, corrected_image)
                
                # Detect errors in correction
                error_map = await self._detect_errors(corrected_image)
                
                # Convert back to image
                corrected_image = self._tensor_to_image(corrected_image)
            else:
                corrected_image = image
                consistency_score = 1.0
                error_map = None
            
            # Save if output path provided
            output_path = None
            if 'output_path' in task:
                output_path = task['output_path']
                corrected_image.save(output_path)
            
            return {
                'status': 'success',
                'output_image': corrected_image,
                'output_path': output_path,
                'quality_score': float(quality_score),
                'critique': critique.tolist() if critique is not None else None,
                'needs_correction': needs_correction,
                'correction_applied': correction_applied,
                'consistency_score': float(consistency_score),
                'error_map': error_map.tolist() if error_map is not None else None
            }
            
        except Exception as e:
            logger.error(f"Meta-correction failed: {e}")
            raise
    
    async def _assess_quality(self, x: torch.Tensor) -> torch.Tensor:
        """Assess image quality"""
        with torch.no_grad():
            return self.models['quality_assessor'](x)
    
    async def _perform_self_critique(self, x: torch.Tensor) -> torch.Tensor:
        """Perform self-critique analysis"""
        with torch.no_grad():
            return self.models['self_critique'](x)
    
    async def _apply_meta_correction(self, x: torch.Tensor) -> torch.Tensor:
        """Apply meta-correction"""
        with torch.no_grad():
            return self.models['meta_correction'](x)
    
    async def _check_consistency(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Check consistency between before and after"""
        with torch.no_grad():
            return self.models['consistency_checker'](before, after)
    
    async def _detect_errors(self, x: torch.Tensor) -> torch.Tensor:
        """Detect errors in corrected image"""
        with torch.no_grad():
            return self.models['error_detector'](x)
    
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