"""
Adaptive Enhancement Agent
Specialized agent for intelligent enhancement based on image characteristics and quality metrics
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

class AdaptiveEnhancementAgent(BaseAgent):
    def __init__(self):
        super().__init__("AdaptiveEnhancementAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.enhancement_types = [
            'sharpness', 'contrast', 'brightness', 'saturation', 'clarity',
            'vibrance', 'highlights', 'shadows', 'midtones', 'color_balance'
        ]
        
    async def _initialize(self) -> None:
        """Initialize adaptive enhancement models"""
        try:
            # Quality assessment network
            self.models['quality_assessor'] = await self._load_quality_assessor()
            
            # Enhancement predictor
            self.models['enhancement_predictor'] = await self._load_enhancement_predictor()
            
            # Adaptive enhancement network
            self.models['adaptive_enhancer'] = await self._load_adaptive_enhancer()
            
            # Multi-scale enhancement network
            self.models['multi_scale_enhancer'] = await self._load_multi_scale_enhancer()
            
            # Quality-aware enhancement network
            self.models['quality_aware_enhancer'] = await self._load_quality_aware_enhancer()
            
            logger.info("Adaptive enhancement models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize adaptive enhancement models: {e}")
            raise
    
    async def _load_quality_assessor(self) -> nn.Module:
        """Load quality assessment network"""
        class QualityAssessor(nn.Module):
            def __init__(self):
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
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.quality_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 10),  # 10 quality metrics
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                features = self.features(x)
                features = features.view(features.size(0), -1)
                quality_scores = self.quality_head(features)
                return quality_scores
        
        return QualityAssessor().to(self.device)
    
    async def _load_enhancement_predictor(self) -> nn.Module:
        """Load enhancement prediction network"""
        class EnhancementPredictor(nn.Module):
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
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                
                self.predictor = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 10),  # 10 enhancement types
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                features = self.features(x)
                features = features.view(features.size(0), -1)
                enhancement_scores = self.predictor(features)
                return enhancement_scores
        
        return EnhancementPredictor().to(self.device)
    
    async def _load_adaptive_enhancer(self) -> nn.Module:
        """Load adaptive enhancement network"""
        class AdaptiveEnhancer(nn.Module):
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
                    nn.ReLU(inplace=True)
                )
                
                self.enhancement_heads = nn.ModuleList([
                    nn.Sequential(
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
                        nn.Tanh()
                    ) for _ in range(10)  # 10 enhancement types
                ])
            
            def forward(self, x, enhancement_weights):
                encoded = self.encoder(x)
                
                # Apply weighted combination of enhancement heads
                enhanced = torch.zeros_like(x)
                for i, head in enumerate(self.enhancement_heads):
                    enhancement = head(encoded)
                    weight = enhancement_weights[:, i:i+1, :, :]
                    enhanced += weight * enhancement
                
                return enhanced
        
        return AdaptiveEnhancer().to(self.device)
    
    async def _load_multi_scale_enhancer(self) -> nn.Module:
        """Load multi-scale enhancement network"""
        class MultiScaleEnhancer(nn.Module):
            def __init__(self):
                super().__init__()
                # Multi-scale feature extraction
                self.scale1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                self.scale2 = nn.Sequential(
                    nn.Conv2d(3, 64, 5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 5, padding=2),
                    nn.ReLU(inplace=True)
                )
                
                self.scale3 = nn.Sequential(
                    nn.Conv2d(3, 64, 7, padding=3),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 7, padding=3),
                    nn.ReLU(inplace=True)
                )
                
                # Feature fusion
                self.fusion = nn.Sequential(
                    nn.Conv2d(192, 128, 3, padding=1),  # 64*3 = 192
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True)
                )
                
                # Enhancement decoder
                self.decoder = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    
                    nn.Conv2d(64, 3, 3, padding=1),
                    nn.Tanh()
                )
            
            def forward(self, x):
                # Multi-scale feature extraction
                features1 = self.scale1(x)
                features2 = self.scale2(x)
                features3 = self.scale3(x)
                
                # Feature fusion
                fused = torch.cat([features1, features2, features3], dim=1)
                fused = self.fusion(fused)
                
                # Enhancement
                enhanced = self.decoder(fused)
                return enhanced
        
        return MultiScaleEnhancer().to(self.device)
    
    async def _load_quality_aware_enhancer(self) -> nn.Module:
        """Load quality-aware enhancement network"""
        class QualityAwareEnhancer(nn.Module):
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
                
                self.quality_conditioning = nn.Sequential(
                    nn.Linear(10, 64),  # 10 quality metrics
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 128),
                    nn.ReLU(inplace=True)
                )
                
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256 + 128, 128, 4, stride=2, padding=1),
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
            
            def forward(self, x, quality_scores):
                encoded = self.encoder(x)
                quality_features = self.quality_conditioning(quality_scores)
                
                # Reshape quality features to match spatial dimensions
                batch_size, channels, height, width = encoded.shape
                quality_features = quality_features.unsqueeze(-1).unsqueeze(-1)
                quality_features = quality_features.expand(-1, -1, height, width)
                
                # Fuse features
                fused = torch.cat([encoded, quality_features], dim=1)
                enhanced = self.decoder(fused)
                return enhanced
        
        return QualityAwareEnhancer().to(self.device)
    
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process adaptive enhancement task"""
        try:
            # Get input image
            image = task.get('image')
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to tensor
            x = self._image_to_tensor(image).unsqueeze(0).to(self.device)
            
            # Assess image quality
            quality_scores = await self._assess_quality(x)
            
            # Predict enhancement needs
            enhancement_scores = await self._predict_enhancement(x)
            
            # Apply adaptive enhancement
            adaptive_enhanced = await self._apply_adaptive_enhancement(x, enhancement_scores)
            
            # Apply multi-scale enhancement
            multi_scale_enhanced = await self._apply_multi_scale_enhancement(adaptive_enhanced)
            
            # Apply quality-aware enhancement
            final_enhanced = await self._apply_quality_aware_enhancement(multi_scale_enhanced, quality_scores)
            
            # Convert back to image
            output_image = self._tensor_to_image(final_enhanced)
            
            # Generate enhancement analysis
            enhancement_analysis = self._analyze_enhancement(quality_scores, enhancement_scores)
            
            # Save if output path provided
            output_path = None
            if 'output_path' in task:
                output_path = task['output_path']
                output_image.save(output_path)
            
            return {
                'status': 'success',
                'output_image': output_image,
                'output_path': output_path,
                'quality_scores': quality_scores.tolist(),
                'enhancement_scores': enhancement_scores.tolist(),
                'enhancement_analysis': enhancement_analysis,
                'enhancement_methods': ['adaptive', 'multi_scale', 'quality_aware']
            }
            
        except Exception as e:
            logger.error(f"Adaptive enhancement failed: {e}")
            raise
    
    async def _assess_quality(self, x: torch.Tensor) -> torch.Tensor:
        """Assess image quality"""
        with torch.no_grad():
            return self.models['quality_assessor'](x)
    
    async def _predict_enhancement(self, x: torch.Tensor) -> torch.Tensor:
        """Predict enhancement needs"""
        with torch.no_grad():
            return self.models['enhancement_predictor'](x)
    
    async def _apply_adaptive_enhancement(self, x: torch.Tensor, enhancement_scores: torch.Tensor) -> torch.Tensor:
        """Apply adaptive enhancement"""
        with torch.no_grad():
            # Reshape enhancement scores for spatial application
            enhancement_weights = enhancement_scores.unsqueeze(-1).unsqueeze(-1)
            enhancement_weights = enhancement_weights.expand(-1, -1, x.shape[2], x.shape[3])
            return self.models['adaptive_enhancer'](x, enhancement_weights)
    
    async def _apply_multi_scale_enhancement(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale enhancement"""
        with torch.no_grad():
            return self.models['multi_scale_enhancer'](x)
    
    async def _apply_quality_aware_enhancement(self, x: torch.Tensor, quality_scores: torch.Tensor) -> torch.Tensor:
        """Apply quality-aware enhancement"""
        with torch.no_grad():
            return self.models['quality_aware_enhancer'](x, quality_scores)
    
    def _analyze_enhancement(self, quality_scores: torch.Tensor, enhancement_scores: torch.Tensor) -> Dict[str, Any]:
        """Analyze enhancement results"""
        quality = quality_scores.squeeze().cpu().numpy()
        enhancement = enhancement_scores.squeeze().cpu().numpy()
        
        analysis = {
            'overall_quality': float(np.mean(quality)),
            'enhancement_applied': [],
            'quality_improvements': {}
        }
        
        # Analyze which enhancements were applied
        for i, enhancement_type in enumerate(self.enhancement_types):
            if enhancement[i] > 0.3:
                analysis['enhancement_applied'].append({
                    'type': enhancement_type,
                    'intensity': float(enhancement[i]),
                    'quality_metric': float(quality[i]) if i < len(quality) else 0.0
                })
        
        # Calculate quality improvements
        for i, enhancement_type in enumerate(self.enhancement_types):
            if i < len(quality):
                analysis['quality_improvements'][enhancement_type] = {
                    'before': float(quality[i]),
                    'enhancement_intensity': float(enhancement[i]) if i < len(enhancement) else 0.0
                }
        
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