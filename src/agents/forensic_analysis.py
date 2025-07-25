"""
Forensic Analysis Agent
Specialized agent for scientific examination and evidence-based restoration decisions
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

class ForensicAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("ForensicAnalysisAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.analysis_methods = [
            'pixel_analysis', 'noise_fingerprinting', 'compression_artifacts',
            'tampering_detection', 'metadata_analysis', 'color_forensics',
            'lighting_analysis', 'perspective_forensics', 'material_forensics'
        ]
        
    async def _initialize(self) -> None:
        """Initialize forensic analysis models"""
        try:
            # Pixel-level analysis network
            self.models['pixel_analyzer'] = await self._load_pixel_analyzer()
            
            # Noise fingerprinting network
            self.models['noise_fingerprinter'] = await self._load_noise_fingerprinter()
            
            # Compression artifact detector
            self.models['compression_detector'] = await self._load_compression_detector()
            
            # Tampering detection network
            self.models['tampering_detector'] = await self._load_tampering_detector()
            
            # Color forensic analyzer
            self.models['color_forensics'] = await self._load_color_forensics()
            
            # Lighting analysis network
            self.models['lighting_analyzer'] = await self._load_lighting_analyzer()
            
            # Perspective forensic network
            self.models['perspective_forensics'] = await self._load_perspective_forensics()
            
            logger.info("Forensic analysis models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize forensic analysis models: {e}")
            raise
    
    async def _load_pixel_analyzer(self) -> nn.Module:
        """Load pixel-level analysis network"""
        class PixelAnalyzer(nn.Module):
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
                
                self.analyzer = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 20)  # 20 pixel analysis features
                )
            
            def forward(self, x):
                features = self.features(x)
                analysis = self.analyzer(features)
                return analysis
        
        return PixelAnalyzer().to(self.device)
    
    async def _load_noise_fingerprinter(self) -> nn.Module:
        """Load noise fingerprinting network"""
        class NoiseFingerprinter(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    nn.Conv2d(64, 128, 5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    nn.Conv2d(128, 256, 5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 5, padding=2),
                    nn.ReLU(inplace=True)
                )
                
                self.fingerprinter = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 32)  # 32 noise fingerprint features
                )
            
            def forward(self, x):
                features = self.features(x)
                fingerprint = self.fingerprinter(features)
                return fingerprint
        
        return NoiseFingerprinter().to(self.device)
    
    async def _load_compression_detector(self) -> nn.Module:
        """Load compression artifact detector"""
        class CompressionDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 8, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 8, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    nn.Conv2d(64, 128, 8, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 8, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, stride=2),
                    
                    nn.Conv2d(128, 256, 8, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 8, stride=1, padding=0),
                    nn.ReLU(inplace=True)
                )
                
                self.detector = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 16)  # 16 compression artifact features
                )
            
            def forward(self, x):
                features = self.features(x)
                artifacts = self.detector(features)
                return artifacts
        
        return CompressionDetector().to(self.device)
    
    async def _load_tampering_detector(self) -> nn.Module:
        """Load tampering detection network"""
        class TamperingDetector(nn.Module):
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
                
                self.tampering_head = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 1, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                features = self.features(x)
                tampering_map = self.tampering_head(features)
                return tampering_map
        
        return TamperingDetector().to(self.device)
    
    async def _load_color_forensics(self) -> nn.Module:
        """Load color forensic analyzer"""
        class ColorForensics(nn.Module):
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
                
                self.color_analyzer = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 24)  # 24 color forensic features
                )
            
            def forward(self, x):
                features = self.features(x)
                color_analysis = self.color_analyzer(features)
                return color_analysis
        
        return ColorForensics().to(self.device)
    
    async def _load_lighting_analyzer(self) -> nn.Module:
        """Load lighting analysis network"""
        class LightingAnalyzer(nn.Module):
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
                
                self.lighting_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 12)  # 12 lighting analysis features
                )
            
            def forward(self, x):
                features = self.features(x)
                lighting = self.lighting_head(features)
                return lighting
        
        return LightingAnalyzer().to(self.device)
    
    async def _load_perspective_forensics(self) -> nn.Module:
        """Load perspective forensic network"""
        class PerspectiveForensics(nn.Module):
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
                
                self.perspective_head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 8)  # 8 perspective forensic features
                )
            
            def forward(self, x):
                features = self.features(x)
                perspective = self.perspective_head(features)
                return perspective
        
        return PerspectiveForensics().to(self.device)
    
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process forensic analysis task"""
        try:
            # Get input image
            image = task.get('image')
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to tensor
            x = self._image_to_tensor(image).unsqueeze(0).to(self.device)
            
            # Perform comprehensive forensic analysis
            pixel_analysis = await self._analyze_pixels(x)
            noise_fingerprint = await self._fingerprint_noise(x)
            compression_artifacts = await self._detect_compression(x)
            tampering_map = await self._detect_tampering(x)
            color_forensics = await self._analyze_color_forensics(x)
            lighting_analysis = await self._analyze_lighting(x)
            perspective_forensics = await self._analyze_perspective(x)
            
            # Generate forensic report
            forensic_report = self._generate_forensic_report(
                pixel_analysis, noise_fingerprint, compression_artifacts,
                tampering_map, color_forensics, lighting_analysis, perspective_forensics
            )
            
            # Determine restoration recommendations
            recommendations = self._generate_recommendations(forensic_report)
            
            return {
                'status': 'success',
                'forensic_report': forensic_report,
                'recommendations': recommendations,
                'pixel_analysis': pixel_analysis.tolist(),
                'noise_fingerprint': noise_fingerprint.tolist(),
                'compression_artifacts': compression_artifacts.tolist(),
                'tampering_map': tampering_map.tolist() if tampering_map is not None else None,
                'color_forensics': color_forensics.tolist(),
                'lighting_analysis': lighting_analysis.tolist(),
                'perspective_forensics': perspective_forensics.tolist()
            }
            
        except Exception as e:
            logger.error(f"Forensic analysis failed: {e}")
            raise
    
    async def _analyze_pixels(self, x: torch.Tensor) -> torch.Tensor:
        """Analyze pixel-level characteristics"""
        with torch.no_grad():
            return self.models['pixel_analyzer'](x)
    
    async def _fingerprint_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Generate noise fingerprint"""
        with torch.no_grad():
            return self.models['noise_fingerprinter'](x)
    
    async def _detect_compression(self, x: torch.Tensor) -> torch.Tensor:
        """Detect compression artifacts"""
        with torch.no_grad():
            return self.models['compression_detector'](x)
    
    async def _detect_tampering(self, x: torch.Tensor) -> torch.Tensor:
        """Detect image tampering"""
        with torch.no_grad():
            return self.models['tampering_detector'](x)
    
    async def _analyze_color_forensics(self, x: torch.Tensor) -> torch.Tensor:
        """Analyze color forensics"""
        with torch.no_grad():
            return self.models['color_forensics'](x)
    
    async def _analyze_lighting(self, x: torch.Tensor) -> torch.Tensor:
        """Analyze lighting characteristics"""
        with torch.no_grad():
            return self.models['lighting_analyzer'](x)
    
    async def _analyze_perspective(self, x: torch.Tensor) -> torch.Tensor:
        """Analyze perspective forensics"""
        with torch.no_grad():
            return self.models['perspective_forensics'](x)
    
    def _generate_forensic_report(self, pixel_analysis: torch.Tensor,
                                 noise_fingerprint: torch.Tensor,
                                 compression_artifacts: torch.Tensor,
                                 tampering_map: torch.Tensor,
                                 color_forensics: torch.Tensor,
                                 lighting_analysis: torch.Tensor,
                                 perspective_forensics: torch.Tensor) -> Dict[str, Any]:
        """Generate comprehensive forensic report"""
        report = {
            'image_authenticity': self._assess_authenticity(tampering_map),
            'compression_analysis': self._analyze_compression_level(compression_artifacts),
            'noise_characteristics': self._analyze_noise_pattern(noise_fingerprint),
            'color_consistency': self._assess_color_consistency(color_forensics),
            'lighting_consistency': self._assess_lighting_consistency(lighting_analysis),
            'perspective_analysis': self._analyze_perspective_consistency(perspective_forensics),
            'pixel_integrity': self._assess_pixel_integrity(pixel_analysis),
            'overall_confidence': self._calculate_overall_confidence(
                pixel_analysis, noise_fingerprint, compression_artifacts,
                tampering_map, color_forensics, lighting_analysis, perspective_forensics
            )
        }
        return report
    
    def _assess_authenticity(self, tampering_map: torch.Tensor) -> Dict[str, Any]:
        """Assess image authenticity"""
        if tampering_map is None:
            return {'score': 1.0, 'status': 'no_tampering_detected'}
        
        tampering_score = float(torch.mean(tampering_map))
        return {
            'score': 1.0 - tampering_score,
            'status': 'authentic' if tampering_score < 0.1 else 'suspicious',
            'tampering_probability': tampering_score
        }
    
    def _analyze_compression_level(self, compression_artifacts: torch.Tensor) -> Dict[str, Any]:
        """Analyze compression level"""
        compression_score = float(torch.mean(compression_artifacts))
        return {
            'compression_level': compression_score,
            'quality_impact': 'high' if compression_score > 0.7 else 'medium' if compression_score > 0.3 else 'low'
        }
    
    def _analyze_noise_pattern(self, noise_fingerprint: torch.Tensor) -> Dict[str, Any]:
        """Analyze noise pattern"""
        noise_features = noise_fingerprint.squeeze().cpu().numpy()
        return {
            'noise_type': self._classify_noise_type(noise_features),
            'noise_level': float(torch.mean(noise_fingerprint)),
            'noise_characteristics': noise_features.tolist()
        }
    
    def _assess_color_consistency(self, color_forensics: torch.Tensor) -> Dict[str, Any]:
        """Assess color consistency"""
        color_score = float(torch.mean(color_forensics))
        return {
            'consistency_score': color_score,
            'status': 'consistent' if color_score > 0.7 else 'inconsistent'
        }
    
    def _assess_lighting_consistency(self, lighting_analysis: torch.Tensor) -> Dict[str, Any]:
        """Assess lighting consistency"""
        lighting_score = float(torch.mean(lighting_analysis))
        return {
            'consistency_score': lighting_score,
            'status': 'consistent' if lighting_score > 0.7 else 'inconsistent'
        }
    
    def _analyze_perspective_consistency(self, perspective_forensics: torch.Tensor) -> Dict[str, Any]:
        """Analyze perspective consistency"""
        perspective_score = float(torch.mean(perspective_forensics))
        return {
            'consistency_score': perspective_score,
            'status': 'consistent' if perspective_score > 0.7 else 'inconsistent'
        }
    
    def _assess_pixel_integrity(self, pixel_analysis: torch.Tensor) -> Dict[str, Any]:
        """Assess pixel integrity"""
        integrity_score = float(torch.mean(pixel_analysis))
        return {
            'integrity_score': integrity_score,
            'status': 'good' if integrity_score > 0.7 else 'fair' if integrity_score > 0.4 else 'poor'
        }
    
    def _calculate_overall_confidence(self, *analyses) -> float:
        """Calculate overall confidence score"""
        scores = [float(torch.mean(analysis)) for analysis in analyses if analysis is not None]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _classify_noise_type(self, noise_features: np.ndarray) -> str:
        """Classify noise type based on features"""
        # Simple classification based on feature patterns
        if np.max(noise_features) > 0.8:
            return 'high_frequency'
        elif np.mean(noise_features) > 0.5:
            return 'medium_frequency'
        else:
            return 'low_frequency'
    
    def _generate_recommendations(self, forensic_report: Dict[str, Any]) -> List[str]:
        """Generate restoration recommendations based on forensic analysis"""
        recommendations = []
        
        if forensic_report['image_authenticity']['status'] == 'suspicious':
            recommendations.append("Image shows signs of tampering - proceed with caution")
        
        if forensic_report['compression_analysis']['quality_impact'] == 'high':
            recommendations.append("High compression artifacts detected - apply compression artifact removal")
        
        if forensic_report['noise_characteristics']['noise_level'] > 0.5:
            recommendations.append("Significant noise detected - apply noise reduction")
        
        if forensic_report['color_consistency']['status'] == 'inconsistent':
            recommendations.append("Color inconsistencies detected - apply color correction")
        
        if forensic_report['lighting_consistency']['status'] == 'inconsistent':
            recommendations.append("Lighting inconsistencies detected - apply lighting correction")
        
        if forensic_report['pixel_integrity']['status'] == 'poor':
            recommendations.append("Poor pixel integrity - apply pixel-level restoration")
        
        return recommendations
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor"""
        # Resize to standard size
        image = image.resize((512, 512))
        # Convert to tensor
        x = torch.from_numpy(np.array(image)).float() / 255.0
        x = x.permute(2, 0, 1)  # HWC to CHW
        return x
    
    async def _cleanup(self) -> None:
        """Cleanup models and resources"""
        self.models.clear()
        torch.cuda.empty_cache() 