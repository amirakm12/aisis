"""
Tile Stitching & Seam Fusion Agent
Handles tiling, overlapping, feathering, and seamless blending of image segments
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

class TileStitchingAgent(BaseAgent):
    def __init__(self):
        super().__init__("TileStitchingAgent")
        self.device = gpu_manager.device
        self.models = {}
        self.transforms = None
        
    async def _initialize(self) -> None:
        """Initialize tile stitching models"""
        try:
            # TODO: Replace with real tile stitching models
            logger.warning("Tile stitching models are placeholders. Implement real models.")
            
            # Seam detection model
            self.models['seam_detector'] = await self._load_seam_detector()
            
            # Feathering model
            self.models['feathering'] = await self._load_feathering_model()
            
            # Blend model
            self.models['blend'] = await self._load_blend_model()
            
            # Overlap detection
            self.models['overlap_detector'] = await self._load_overlap_detector()
            
            # Setup transforms
            self.transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Tile stitching models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize tile stitching models: {e}")
            raise
    
    async def _load_seam_detector(self) -> nn.Module:
        """Load seam detection model"""
        class SeamDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 3, padding=1)
            
            def forward(self, x):
                return torch.sigmoid(self.conv(x))
        
        return SeamDetector().to(self.device)
    
    async def _load_feathering_model(self) -> nn.Module:
        """Load feathering model"""
        class Feathering(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 5, padding=2)
            
            def forward(self, x):
                return self.conv(x)
        
        return Feathering().to(self.device)
    
    async def _load_blend_model(self) -> nn.Module:
        """Load blend model"""
        class Blend(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(6, 3, 3, padding=1)  # 2 images + mask
        
            def forward(self, x1, x2, mask):
                x = torch.cat([x1, x2], dim=1)
                return self.conv(x)
        
        return Blend().to(self.device)
    
    async def _load_overlap_detector(self) -> nn.Module:
        """Load overlap detection model"""
        class OverlapDetector(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 1, 3, padding=1)
            
            def forward(self, x):
                return torch.sigmoid(self.conv(x))
        
        return OverlapDetector().to(self.device)
    
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process tile stitching task"""
        try:
            # Get input tiles
            tiles = task.get('tiles', [])
            if not tiles:
                raise ValueError("No tiles provided for stitching")
            
            # Get stitching parameters
            overlap = task.get('overlap', 64)
            feather_width = task.get('feather_width', 32)
            
            # Convert tiles to tensors
            tile_tensors = []
            for tile_path in tiles:
                if isinstance(tile_path, str):
                    tile = Image.open(tile_path).convert('RGB')
                elif isinstance(tile_path, np.ndarray):
                    tile = Image.fromarray(tile_path)
                else:
                    tile = tile_path
                
                tile_tensor = T.ToTensor()(tile).to(self.device)
                tile_tensors.append(tile_tensor)
            
            # Stitch tiles
            stitched = await self._stitch_tiles(tile_tensors, overlap, feather_width)
            
            # Convert back to image
            output_image = self._tensor_to_image(stitched)
            
            # Save if output path provided
            output_path = None
            if 'output_path' in task:
                output_path = task['output_path']
                output_image.save(output_path)
            
            return {
                'status': 'success',
                'output_image': output_image,
                'output_path': output_path,
                'num_tiles': len(tiles),
                'overlap': overlap,
                'feather_width': feather_width
            }
            
        except Exception as e:
            logger.error(f"Tile stitching failed: {e}")
            raise
    
    async def _stitch_tiles(self, tiles: List[torch.Tensor], overlap: int, feather_width: int) -> torch.Tensor:
        """Stitch multiple tiles together"""
        if len(tiles) == 1:
            return tiles[0]
        
        # Simple grid stitching for now
        # TODO: Implement intelligent tile arrangement detection
        
        # Assume tiles are in a grid pattern
        grid_size = int(np.sqrt(len(tiles)))
        if grid_size * grid_size != len(tiles):
            # Fallback to horizontal stitching
            return await self._stitch_horizontal(tiles, overlap, feather_width)
        
        # Stitch in grid pattern
        rows = []
        for i in range(grid_size):
            row_tiles = tiles[i * grid_size:(i + 1) * grid_size]
            row = await self._stitch_horizontal(row_tiles, overlap, feather_width)
            rows.append(row)
        
        # Stitch rows vertically
        return await self._stitch_vertical(rows, overlap, feather_width)
    
    async def _stitch_horizontal(self, tiles: List[torch.Tensor], overlap: int, feather_width: int) -> torch.Tensor:
        """Stitch tiles horizontally"""
        if len(tiles) == 1:
            return tiles[0]
        
        result = tiles[0]
        for i in range(1, len(tiles)):
            result = await self._blend_tiles(result, tiles[i], overlap, feather_width, 'horizontal')
        
        return result
    
    async def _stitch_vertical(self, tiles: List[torch.Tensor], overlap: int, feather_width: int) -> torch.Tensor:
        """Stitch tiles vertically"""
        if len(tiles) == 1:
            return tiles[0]
        
        result = tiles[0]
        for i in range(1, len(tiles)):
            result = await self._blend_tiles(result, tiles[i], overlap, feather_width, 'vertical')
        
        return result
    
    async def _blend_tiles(self, tile1: torch.Tensor, tile2: torch.Tensor, overlap: int, feather_width: int, direction: str) -> torch.Tensor:
        """Blend two tiles with overlap"""
        # Create feathering mask
        mask = await self._create_feathering_mask(tile1.shape, overlap, feather_width, direction)
        
        # Apply feathering
        feathered1 = await self._apply_feathering(tile1, mask)
        feathered2 = await self._apply_feathering(tile2, 1 - mask)
        
        # Blend tiles
        blended = await self._blend_with_model(feathered1, feathered2, mask)
        
        return blended
    
    async def _create_feathering_mask(self, shape: Tuple[int, ...], overlap: int, feather_width: int, direction: str) -> torch.Tensor:
        """Create feathering mask for smooth blending"""
        _, _, h, w = shape
        
        if direction == 'horizontal':
            mask = torch.zeros(1, 1, h, w)
            mask[:, :, :, :overlap] = torch.linspace(0, 1, overlap).view(1, 1, 1, -1)
        else:  # vertical
            mask = torch.zeros(1, 1, h, w)
            mask[:, :, :overlap, :] = torch.linspace(0, 1, overlap).view(1, 1, -1, 1)
        
        return mask.to(self.device)
    
    async def _apply_feathering(self, tile: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply feathering to tile"""
        with torch.no_grad():
            return self.models['feathering'](tile) * mask
    
    async def _blend_with_model(self, tile1: torch.Tensor, tile2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Blend tiles using the blend model"""
        with torch.no_grad():
            return self.models['blend'](tile1, tile2, mask)
    
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