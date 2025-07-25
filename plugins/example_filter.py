"""
Example Image Filter Plugin for AISIS
Demonstrates how to create a simple image processing plugin
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Any, Optional

from src.plugins.base_plugin import BasePlugin, PluginMetadata

class ExampleFilterPlugin(BasePlugin):
    """
    Example plugin that provides various image filters
    """
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_filter",
            version="1.0.0",
            description="Example image filter plugin with various effects",
            author="AISIS Team",
            license="MIT",
            tags=["image", "filter", "effects", "example"],
            dependencies=["Pillow", "numpy"]
        )
    
    def initialize(self) -> bool:
        """Initialize the plugin"""
        self.filters = {
            'blur': self._apply_blur,
            'sharpen': self._apply_sharpen,
            'edge_enhance': self._apply_edge_enhance,
            'emboss': self._apply_emboss,
            'brightness': self._adjust_brightness,
            'contrast': self._adjust_contrast,
            'saturation': self._adjust_saturation,
            'vintage': self._apply_vintage_effect,
            'sepia': self._apply_sepia,
            'black_and_white': self._apply_bw
        }
        
        self.logger.info("Example Filter Plugin initialized with filters: " + 
                        ", ".join(self.filters.keys()))
        return True
    
    def execute(self, image: Image.Image, operation: str, **parameters) -> Image.Image:
        """
        Execute a filter operation on an image
        
        Args:
            image: PIL Image to process
            operation: Filter operation to apply
            **parameters: Operation-specific parameters
            
        Returns:
            Processed PIL Image
        """
        if operation not in self.filters:
            available = ", ".join(self.filters.keys())
            raise ValueError(f"Unknown operation '{operation}'. Available: {available}")
        
        try:
            result = self.filters[operation](image, **parameters)
            self.logger.info(f"Applied filter '{operation}' successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to apply filter '{operation}': {e}")
            raise
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get plugin capabilities"""
        return {
            'operations': list(self.filters.keys()),
            'input_formats': ['JPEG', 'PNG', 'TIFF', 'BMP'],
            'output_formats': ['JPEG', 'PNG', 'TIFF', 'BMP'],
            'parameters': {
                'blur': {'radius': 'float (default: 2.0)'},
                'brightness': {'factor': 'float (default: 1.2)'},
                'contrast': {'factor': 'float (default: 1.2)'},
                'saturation': {'factor': 'float (default: 1.2)'},
                'vintage': {'intensity': 'float (default: 0.5)'},
                'sepia': {'intensity': 'float (default: 1.0)'}
            }
        }
    
    def _apply_blur(self, image: Image.Image, radius: float = 2.0) -> Image.Image:
        """Apply Gaussian blur"""
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _apply_sharpen(self, image: Image.Image) -> Image.Image:
        """Apply sharpening filter"""
        return image.filter(ImageFilter.SHARPEN)
    
    def _apply_edge_enhance(self, image: Image.Image) -> Image.Image:
        """Apply edge enhancement"""
        return image.filter(ImageFilter.EDGE_ENHANCE)
    
    def _apply_emboss(self, image: Image.Image) -> Image.Image:
        """Apply emboss effect"""
        return image.filter(ImageFilter.EMBOSS)
    
    def _adjust_brightness(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Adjust image brightness"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def _adjust_contrast(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Adjust image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def _adjust_saturation(self, image: Image.Image, factor: float = 1.2) -> Image.Image:
        """Adjust color saturation"""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def _apply_vintage_effect(self, image: Image.Image, intensity: float = 0.5) -> Image.Image:
        """Apply vintage/retro effect"""
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply sepia tone
        sepia_image = self._apply_sepia(image, intensity)
        
        # Reduce brightness slightly
        brightness_enhancer = ImageEnhance.Brightness(sepia_image)
        darker = brightness_enhancer.enhance(0.9)
        
        # Reduce contrast slightly
        contrast_enhancer = ImageEnhance.Contrast(darker)
        result = contrast_enhancer.enhance(0.8)
        
        return result
    
    def _apply_sepia(self, image: Image.Image, intensity: float = 1.0) -> Image.Image:
        """Apply sepia tone effect"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Sepia transformation matrix
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        # Apply sepia transformation
        sepia_img = img_array @ sepia_matrix.T
        sepia_img = np.clip(sepia_img, 0, 255)
        
        # Blend with original based on intensity
        result_array = (1 - intensity) * img_array + intensity * sepia_img
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result_array)
    
    def _apply_bw(self, image: Image.Image) -> Image.Image:
        """Convert to black and white"""
        return image.convert('L').convert('RGB')
    
    def cleanup(self):
        """Cleanup plugin resources"""
        self.filters.clear()
        self.logger.info("Example Filter Plugin cleaned up")

# Plugin class must be available at module level
Plugin = ExampleFilterPlugin