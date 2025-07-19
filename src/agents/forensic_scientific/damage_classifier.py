"""
Damage Classifier Agent
Automatically detects regions with dirt, mold, tears, water damage and directs the right AI repair strategy
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

class DamageClassifierAgent(BaseAgent):
    def __init__(self):
        super().__init__("DamageClassifierAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.transforms = None
        self.damage_types = [
            'dirt', 'mold', 'tears', 'water_damage', 'scratches', 'fading',
            'stains', 'burn_damage', 'acid_damage', 'foxing', 'insect_damage',
            'adhesive_residue', 'yellowing', 'brittleness', 'missing_parts'
        ]
        
    async def _initialize(self) -> None:
        """Initialize damage classification models"""
        try:
            # TODO: Replace with real damage classification models
            logger.warning("Damage classification models are placeholders. Implement real models.")
            
            # Damage classifier
            self.models['damage_classifier'] = await self._load_damage_classifier()
            
            # Damage segmentation
            self.models['damage_segmentation'] = await self._load_damage_segmentation()
            
            # Severity assessment
            self.models['severity_assessor'] = await self._load_severity_assessor()
            
            # Repair strategy selector
            self.models['repair_strategy'] = await self._load_repair_strategy()
            
            # Setup transforms
            self.transforms = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Damage classification models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize damage classification models: {e}")
            raise
    
    async def _load_damage_classifier(self) -> nn.Module:
        """Load damage classification model"""
        class DamageClassifier(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1)
                )
                self.classifier = nn.Linear(128, num_classes)
            
            def forward(self, x):
                features = self.backbone(x)
                features = features.squeeze(-1).squeeze(-1)
                return torch.sigmoid(self.classifier(features))
        
        return DamageClassifier(len(self.damage_types)).to(self.device)
    
    async def _load_damage_segmentation(self) -> nn.Module:
        """Load damage segmentation model"""
        class DamageSegmentation(nn.Module):
            def __init__(self, num_classes: int):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.Conv2d(64, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, num_classes, 1)
                )
            
            def forward(self, x):
                x = self.encoder(x)
                return torch.sigmoid(self.decoder(x))
        
        return DamageSegmentation(len(self.damage_types)).to(self.device)
    
    async def _load_severity_assessor(self) -> nn.Module:
        """Load damage severity assessment model"""
        class SeverityAssessor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 5)  # Severity levels: none, mild, moderate, severe, critical
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).squeeze(-1).squeeze(-1)
                return torch.softmax(self.fc(x), dim=1)
        
        return SeverityAssessor().to(self.device)
    
    async def _load_repair_strategy(self) -> nn.Module:
        """Load repair strategy selection model"""
        class RepairStrategy(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, 20)  # Repair strategies
            
            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = self.pool(x).squeeze(-1).squeeze(-1)
                return torch.sigmoid(self.fc(x))
        
        return RepairStrategy().to(self.device)
    
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process damage classification task"""
        try:
            # Get input image
            image = task.get('image')
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Convert to tensor
            x = self.transforms(image).unsqueeze(0).to(self.device)
            
            # Get task type
            task_type = task.get('task_type', 'full_analysis')
            
            if task_type == 'classify':
                result = await self._classify_damages(x)
            elif task_type == 'segment':
                result = await self._segment_damages(x)
            elif task_type == 'assess_severity':
                result = await self._assess_severity(x)
            else:
                # Full analysis
                result = await self._full_damage_analysis(x, image)
            
            return result
            
        except Exception as e:
            logger.error(f"Damage classification failed: {e}")
            raise
    
    async def _classify_damages(self, x: torch.Tensor) -> Dict[str, Any]:
        """Classify types of damage in image"""
        with torch.no_grad():
            damage_probs = self.models['damage_classifier'](x)
        
        # Get detected damages
        detected_damages = []
        for i, prob in enumerate(damage_probs[0]):
            if prob.item() > 0.5:  # Threshold for detection
                detected_damages.append({
                    'damage_type': self.damage_types[i],
                    'confidence': prob.item()
                })
        
        # Sort by confidence
        detected_damages.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'status': 'success',
            'detected_damages': detected_damages,
            'damage_probabilities': damage_probs.cpu().numpy()
        }
    
    async def _segment_damages(self, x: torch.Tensor) -> Dict[str, Any]:
        """Segment damage regions in image"""
        with torch.no_grad():
            damage_masks = self.models['damage_segmentation'](x)
        
        # Convert to numpy for processing
        masks_np = damage_masks.squeeze(0).cpu().numpy()
        
        # Create damage map
        damage_map = np.zeros((masks_np.shape[1], masks_np.shape[2]), dtype=np.uint8)
        damage_regions = []
        
        for i, mask in enumerate(masks_np):
            if mask.max() > 0.3:  # Threshold for significant damage
                damage_map = np.maximum(damage_map, (mask > 0.3).astype(np.uint8) * (i + 1))
                damage_regions.append({
                    'damage_type': self.damage_types[i],
                    'mask': mask,
                    'area_percentage': (mask > 0.3).sum() / mask.size * 100
                })
        
        return {
            'status': 'success',
            'damage_map': damage_map,
            'damage_regions': damage_regions,
            'total_damage_percentage': (damage_map > 0).sum() / damage_map.size * 100
        }
    
    async def _assess_severity(self, x: torch.Tensor) -> Dict[str, Any]:
        """Assess overall damage severity"""
        with torch.no_grad():
            severity_probs = self.models['severity_assessor'](x)
        
        severity_levels = ['none', 'mild', 'moderate', 'severe', 'critical']
        severity_idx = torch.argmax(severity_probs, dim=1).item()
        
        return {
            'status': 'success',
            'severity_level': severity_levels[severity_idx],
            'severity_confidence': severity_probs[0, severity_idx].item(),
            'severity_probabilities': severity_probs.cpu().numpy()
        }
    
    async def _full_damage_analysis(self, x: torch.Tensor, original_image: Image.Image) -> Dict[str, Any]:
        """Full damage analysis pipeline"""
        # Classify damages
        damage_result = await self._classify_damages(x)
        
        # Segment damage regions
        segmentation_result = await self._segment_damages(x)
        
        # Assess severity
        severity_result = await self._assess_severity(x)
        
        # Get repair strategies
        repair_strategies = await self._get_repair_strategies(x, damage_result, severity_result)
        
        return {
            'status': 'success',
            'detected_damages': damage_result['detected_damages'],
            'damage_regions': segmentation_result['damage_regions'],
            'severity_level': severity_result['severity_level'],
            'repair_strategies': repair_strategies,
            'total_damage_percentage': segmentation_result['total_damage_percentage'],
            'priority_actions': self._get_priority_actions(
                damage_result['detected_damages'],
                severity_result['severity_level']
            )
        }
    
    async def _get_repair_strategies(self, x: torch.Tensor, damage_result: Dict[str, Any], severity_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommended repair strategies"""
        with torch.no_grad():
            strategy_probs = self.models['repair_strategy'](x)
        
        strategies = []
        strategy_names = [
            'gentle_cleaning', 'chemical_treatment', 'mechanical_repair',
            'inpainting', 'consolidation', 'deacidification', 'stabilization',
            'reconstruction', 'protective_coating', 'color_correction',
            'texture_reconstruction', 'structural_repair', 'surface_cleaning',
            'mold_removal', 'stain_removal', 'tear_repair', 'water_damage_treatment',
            'acid_neutralization', 'adhesive_removal', 'pigment_analysis'
        ]
        
        for i, prob in enumerate(strategy_probs[0]):
            if prob.item() > 0.3:  # Threshold for strategy recommendation
                strategies.append({
                    'strategy': strategy_names[i],
                    'confidence': prob.item(),
                    'applicable_damages': self._get_applicable_damages(strategy_names[i])
                })
        
        # Sort by confidence
        strategies.sort(key=lambda x: x['confidence'], reverse=True)
        
        return strategies
    
    def _get_applicable_damages(self, strategy: str) -> List[str]:
        """Get damages that a strategy can address"""
        strategy_damage_map = {
            'gentle_cleaning': ['dirt', 'dust', 'surface_grime'],
            'chemical_treatment': ['mold', 'stains', 'acid_damage'],
            'mechanical_repair': ['tears', 'scratches', 'missing_parts'],
            'inpainting': ['missing_parts', 'tears', 'scratches'],
            'consolidation': ['brittleness', 'flaking'],
            'deacidification': ['acid_damage', 'yellowing'],
            'stabilization': ['brittleness', 'structural_damage'],
            'reconstruction': ['missing_parts', 'severe_damage'],
            'protective_coating': ['surface_protection', 'preventive'],
            'color_correction': ['fading', 'yellowing', 'color_loss'],
            'texture_reconstruction': ['texture_loss', 'surface_damage'],
            'structural_repair': ['tears', 'structural_damage'],
            'surface_cleaning': ['dirt', 'surface_grime'],
            'mold_removal': ['mold', 'fungal_damage'],
            'stain_removal': ['stains', 'water_damage'],
            'tear_repair': ['tears', 'rips'],
            'water_damage_treatment': ['water_damage', 'stains'],
            'acid_neutralization': ['acid_damage', 'foxing'],
            'adhesive_removal': ['adhesive_residue', 'tape_damage'],
            'pigment_analysis': ['fading', 'color_loss', 'pigment_damage']
        }
        
        return strategy_damage_map.get(strategy, [])
    
    def _get_priority_actions(self, detected_damages: List[Dict[str, Any]], severity_level: str) -> List[str]:
        """Get priority actions based on damage analysis"""
        priority_actions = []
        
        # Critical damages first
        critical_damages = ['mold', 'acid_damage', 'water_damage']
        for damage in detected_damages:
            if damage['damage_type'] in critical_damages and damage['confidence'] > 0.7:
                priority_actions.append(f"URGENT: Treat {damage['damage_type']}")
        
        # Structural damages
        structural_damages = ['tears', 'missing_parts', 'brittleness']
        for damage in detected_damages:
            if damage['damage_type'] in structural_damages:
                priority_actions.append(f"High Priority: Repair {damage['damage_type']}")
        
        # Surface damages
        surface_damages = ['dirt', 'stains', 'scratches']
        for damage in detected_damages:
            if damage['damage_type'] in surface_damages:
                priority_actions.append(f"Medium Priority: Clean {damage['damage_type']}")
        
        # Severity-based recommendations
        if severity_level == 'critical':
            priority_actions.insert(0, "CRITICAL: Immediate professional intervention required")
        elif severity_level == 'severe':
            priority_actions.insert(0, "SEVERE: Professional restoration recommended")
        elif severity_level == 'moderate':
            priority_actions.insert(0, "MODERATE: Careful restoration needed")
        
        return priority_actions
    
    async def _cleanup(self) -> None:
        """Cleanup models and resources"""
        self.models.clear()
        torch.cuda.empty_cache() 