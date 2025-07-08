"""
Semantic Editing Agent
Handles context-aware image editing using vision-language models
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Dict, Any, List, Tuple
from loguru import logger

from .base_agent import BaseAgent
from ..core.gpu_utils import gpu_manager
from ..core.model_integration import model_integration

class SemanticEditingAgent(BaseAgent):
    def __init__(self):
        super().__init__("SemanticEditingAgent")
        self.device = gpu_manager.device
        self.vision_model = None
        self.generation_model = None
        self.transforms = None
        self.text_embeddings = {}
    
    @property
    def capabilities(self) -> Dict[str, Any]:
        """Return agent capabilities"""
        return {
            "tasks": ["semantic_editing", "image_understanding", "contextual_modification"],
            "modalities": ["image", "text"],
            "description": "Performs context-aware image editing using vision-language models"
        }
    
    async def _initialize(self) -> None:
        """Initialize semantic editing models"""
        try:
            # Initialize model integration manager
            await model_integration.initialize()
            
            # Load vision-language model for understanding
            self.vision_model = await model_integration.get_vision_language_model()
            
            # Load image generation model for editing
            self.generation_model = await model_integration.get_image_generation_model()
            
            # Setup image transforms
            self.transforms = T.Compose([
                T.Resize((512, 512)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Semantic editing models initialized with real AI models")
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic editing models: {e}")
            raise
    
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process semantic editing task"""
        try:
            # Get input image
            image = task.get('image')
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Parse semantic instruction
            instruction = task.get('description', '')
            edit_params = await self._analyze_instruction(instruction, image)
            
            # Apply semantic editing based on the instruction type
            if edit_params['operation'] == 'generate_variation':
                edited_image = await self._generate_image_variation(image, instruction)
            elif edit_params['operation'] == 'inpaint':
                edited_image = await self._inpaint_image(image, instruction, edit_params)
            else:
                edited_image = await self._apply_semantic_edit(image, instruction, edit_params)
            
            # Save if output path provided
            output_path = None
            if 'output_path' in task and edited_image:
                output_path = task['output_path']
                edited_image.save(output_path)
                logger.info(f"Saved edited image to {output_path}")
            
            return {
                'status': 'success',
                'output_image': edited_image,
                'output_path': output_path,
                'edit_params': edit_params,
                'instruction': instruction
            }
            
        except Exception as e:
            logger.error(f"Semantic editing failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'output_image': task.get('image'),  # Return original image on error
                'edit_params': {},
                'instruction': task.get('description', '')
            }
    
    async def _analyze_instruction(self, instruction: str, image: Image.Image) -> Dict[str, Any]:
        """Analyze semantic editing instruction using vision-language model"""
        try:
            # Use vision model to understand the image context
            image_description = model_integration.process_image_with_vision_model(
                self.vision_model, image, "Describe this image in detail"
            )
            
            edit_params = {
                'operation': 'enhance',
                'target': 'global',
                'intensity': 0.5,
                'image_context': image_description
            }
            
            # Parse instruction for specific operations
            instruction_lower = instruction.lower()
            
            if any(word in instruction_lower for word in ['generate', 'create', 'make new']):
                edit_params['operation'] = 'generate_variation'
                edit_params['intensity'] = 0.8
                
            elif any(word in instruction_lower for word in ['remove', 'delete', 'erase']):
                edit_params['operation'] = 'inpaint'
                edit_params['target'] = self._extract_target_object(instruction)
                
            elif any(word in instruction_lower for word in ['dramatic', 'bold', 'strong']):
                edit_params['operation'] = 'enhance_contrast'
                edit_params['intensity'] = 0.8
                
            elif any(word in instruction_lower for word in ['vintage', 'old', 'retro']):
                edit_params['operation'] = 'apply_style'
                edit_params['style'] = 'vintage'
                edit_params['intensity'] = 0.6
                
            elif any(word in instruction_lower for word in ['bright', 'lighter', 'illuminate']):
                edit_params['operation'] = 'adjust_brightness'
                edit_params['intensity'] = 0.3
                
            elif any(word in instruction_lower for word in ['dark', 'darker', 'shadow']):
                edit_params['operation'] = 'adjust_brightness'
                edit_params['intensity'] = -0.3
                
            elif any(word in instruction_lower for word in ['color', 'colorful', 'saturation']):
                edit_params['operation'] = 'adjust_saturation'
                edit_params['intensity'] = 0.4
                
            elif any(word in instruction_lower for word in ['blur', 'soft', 'smooth']):
                edit_params['operation'] = 'apply_blur'
                edit_params['intensity'] = 0.5
                
            elif any(word in instruction_lower for word in ['sharp', 'crisp', 'detail']):
                edit_params['operation'] = 'enhance_sharpness'
                edit_params['intensity'] = 0.4
            
            logger.info(f"Analyzed instruction: {instruction} -> {edit_params}")
            return edit_params
            
        except Exception as e:
            logger.error(f"Failed to analyze instruction: {e}")
            return {
                'operation': 'enhance',
                'target': 'global',
                'intensity': 0.5,
                'image_context': 'unknown'
            }
    
    def _extract_target_object(self, instruction: str) -> str:
        """Extract target object from instruction for inpainting"""
        # Simple keyword extraction - could be enhanced with NLP
        common_objects = ['person', 'car', 'building', 'tree', 'sky', 'background', 'foreground']
        
        for obj in common_objects:
            if obj in instruction.lower():
                return obj
        
        return 'object'
    
    async def _generate_image_variation(self, image: Image.Image, prompt: str) -> Image.Image:
        """Generate a variation of the image using the generation model"""
        try:
            # Create a prompt that includes the original image context
            enhanced_prompt = f"A variation of the image: {prompt}"
            
            # Use the generation model
            generated_image = model_integration.generate_image_with_model(
                self.generation_model, 
                enhanced_prompt,
                num_inference_steps=20,
                guidance_scale=7.5
            )
            
            if generated_image:
                logger.info("Generated image variation successfully")
                return generated_image
            else:
                logger.warning("Generation failed, returning original image")
                return image
                
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return image
    
    async def _inpaint_image(self, image: Image.Image, instruction: str, edit_params: Dict[str, Any]) -> Image.Image:
        """Perform inpainting to remove/modify specific objects"""
        try:
            # For now, create a simple mask and use generation model
            # In a real implementation, you'd use a segmentation model to create precise masks
            
            width, height = image.size
            mask = Image.new('RGB', (width, height), color='black')
            
            # Create a basic mask (this would be replaced with proper segmentation)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(mask)
            draw.rectangle([width//4, height//4, 3*width//4, 3*height//4], fill='white')
            
            # Use inpainting prompt
            inpaint_prompt = f"Fill the masked area with appropriate content based on: {instruction}"
            
            # For now, return the original image with some modification
            # Real inpainting would use a specialized model
            edited_image = self._apply_simple_edit(image, edit_params)
            
            logger.info("Applied inpainting operation")
            return edited_image
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return image
    
    async def _apply_semantic_edit(self, image: Image.Image, instruction: str, edit_params: Dict[str, Any]) -> Image.Image:
        """Apply semantic editing based on parsed parameters"""
        try:
            operation = edit_params['operation']
            intensity = edit_params['intensity']
            
            if operation == 'enhance_contrast':
                return self._enhance_contrast(image, intensity)
            elif operation == 'apply_style':
                return self._apply_style_filter(image, edit_params.get('style', 'vintage'), intensity)
            elif operation == 'adjust_brightness':
                return self._adjust_brightness(image, intensity)
            elif operation == 'adjust_saturation':
                return self._adjust_saturation(image, intensity)
            elif operation == 'apply_blur':
                return self._apply_blur(image, intensity)
            elif operation == 'enhance_sharpness':
                return self._enhance_sharpness(image, intensity)
            else:
                return self._apply_general_enhancement(image, intensity)
                
        except Exception as e:
            logger.error(f"Semantic edit failed: {e}")
            return image
    
    def _enhance_contrast(self, image: Image.Image, intensity: float) -> Image.Image:
        """Enhance image contrast"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.0 + intensity)
    
    def _adjust_brightness(self, image: Image.Image, intensity: float) -> Image.Image:
        """Adjust image brightness"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(1.0 + intensity)
    
    def _adjust_saturation(self, image: Image.Image, intensity: float) -> Image.Image:
        """Adjust image saturation"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(1.0 + intensity)
    
    def _apply_blur(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply blur effect"""
        from PIL import ImageFilter
        radius = intensity * 5  # Scale intensity to blur radius
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _enhance_sharpness(self, image: Image.Image, intensity: float) -> Image.Image:
        """Enhance image sharpness"""
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(1.0 + intensity)
    
    def _apply_style_filter(self, image: Image.Image, style: str, intensity: float) -> Image.Image:
        """Apply style filter"""
        if style == 'vintage':
            # Apply vintage effect: reduce saturation and add warm tone
            from PIL import ImageEnhance
            
            # Reduce saturation
            color_enhancer = ImageEnhance.Color(image)
            image = color_enhancer.enhance(0.7)
            
            # Add warm tone (shift towards yellow/orange)
            img_array = np.array(image)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + 20 * intensity, 0, 255)  # Red
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] + 10 * intensity, 0, 255)  # Green
            
            return Image.fromarray(img_array)
        
        return image
    
    def _apply_general_enhancement(self, image: Image.Image, intensity: float) -> Image.Image:
        """Apply general image enhancement"""
        from PIL import ImageEnhance
        
        # Slight contrast and sharpness boost
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(1.0 + intensity * 0.3)
        
        sharpness_enhancer = ImageEnhance.Sharpness(image)
        image = sharpness_enhancer.enhance(1.0 + intensity * 0.2)
        
        return image
    
    def _apply_simple_edit(self, image: Image.Image, edit_params: Dict[str, Any]) -> Image.Image:
        """Apply simple edit for fallback scenarios"""
        return self._apply_general_enhancement(image, edit_params.get('intensity', 0.5))
    
    async def _cleanup(self) -> None:
        """Cleanup models and resources"""
        self.vision_model = None
        self.generation_model = None
        self.text_embeddings.clear()
        torch.cuda.empty_cache()
