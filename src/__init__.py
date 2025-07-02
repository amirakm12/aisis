"""
AISIS - AI Creative Studio
Professional-grade image restoration and enhancement system
"""

import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger

from .agents.orchestrator import OrchestratorAgent
from .agents.image_restoration import ImageRestorationAgent
from .agents.style_aesthetic import StyleAestheticAgent
from .agents.semantic_editing import SemanticEditingAgent
from .agents.auto_retouch import AutoRetouchAgent
from .agents.generative import GenerativeAgent
from .agents.neural_radiance import NeuralRadianceAgent
from .agents.denoising import DenoisingAgent
from .agents.super_resolution import SuperResolutionAgent
from .agents.color_correction import ColorCorrectionAgent
from .agents.tile_stitching import TileStitchingAgent
from .agents.text_recovery import TextRecoveryAgent
from .agents.feedback_loop import FeedbackLoopAgent
from .agents.perspective_correction import PerspectiveCorrectionAgent
from .agents.material_recognition import MaterialRecognitionAgent
from .agents.damage_classifier import DamageClassifierAgent
from .agents.hyperspectral_recovery import HyperspectralRecoveryAgent
from .agents.paint_layer_decomposition import PaintLayerDecompositionAgent
from .agents.meta_correction import MetaCorrectionAgent
from .agents.self_critique import SelfCritiqueAgent
from .agents.forensic_analysis import ForensicAnalysisAgent
from .agents.context_aware_restoration import ContextAwareRestorationAgent
from .agents.adaptive_enhancement import AdaptiveEnhancementAgent
from .agents.hyper_orchestrator import HyperOrchestrator

class AISIS:
    """
    AISIS - AI Creative Studio
    Professional-grade image restoration and enhancement system with comprehensive multi-agent architecture
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AISIS with comprehensive restoration capabilities
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.orchestrator = OrchestratorAgent()
        self.agents = {}
        self.is_initialized = False
        
        # Initialize individual agents for direct access
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        # Core restoration agents
        self.agents['image_restoration'] = ImageRestorationAgent()
        self.agents['style_aesthetic'] = StyleAestheticAgent()
        self.agents['semantic_editing'] = SemanticEditingAgent()
        self.agents['auto_retouch'] = AutoRetouchAgent()
        self.agents['generative'] = GenerativeAgent()
        self.agents['neural_radiance'] = NeuralRadianceAgent()
        self.agents['denoising'] = DenoisingAgent()
        self.agents['super_resolution'] = SuperResolutionAgent()
        self.agents['color_correction'] = ColorCorrectionAgent()
        self.agents['tile_stitching'] = TileStitchingAgent()
        self.agents['text_recovery'] = TextRecoveryAgent()
        self.agents['feedback_loop'] = FeedbackLoopAgent()
        self.agents['perspective_correction'] = PerspectiveCorrectionAgent()
        
        # Scientific and forensic agents
        self.agents['material_recognition'] = MaterialRecognitionAgent()
        self.agents['damage_classifier'] = DamageClassifierAgent()
        self.agents['hyperspectral_recovery'] = HyperspectralRecoveryAgent()
        self.agents['paint_layer_decomposition'] = PaintLayerDecompositionAgent()
        self.agents['forensic_analysis'] = ForensicAnalysisAgent()
        
        # Advanced AI agents
        self.agents['meta_correction'] = MetaCorrectionAgent()
        self.agents['self_critique'] = SelfCritiqueAgent()
        self.agents['context_aware_restoration'] = ContextAwareRestorationAgent()
        self.agents['adaptive_enhancement'] = AdaptiveEnhancementAgent()
    
    async def initialize(self):
        """Initialize the AISIS system"""
        try:
            logger.info("Initializing AISIS - AI Creative Studio...")
            
            # Initialize orchestrator
            await self.orchestrator.initialize()
            
            # Initialize individual agents
            for name, agent in self.agents.items():
                logger.info(f"Initializing {name} agent...")
                await agent.initialize()
            
            self.is_initialized = True
            logger.info("AISIS initialized successfully with comprehensive restoration capabilities")
            
        except Exception as e:
            logger.error(f"Failed to initialize AISIS: {e}")
            raise
    
    async def restore_image(self, 
                          image_path: Union[str, Path, Image.Image, np.ndarray],
                          output_path: Optional[str] = None,
                          restoration_type: str = 'comprehensive',
                          **kwargs) -> Dict[str, Any]:
        """
        Perform comprehensive image restoration
        
        Args:
            image_path: Input image path, PIL Image, or numpy array
            output_path: Output path for restored image
            restoration_type: Type of restoration ('comprehensive', 'basic', 'scientific', 'artistic')
            **kwargs: Additional restoration parameters
            
        Returns:
            Dictionary containing restoration results and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Prepare input image
        if isinstance(image_path, (str, Path)):
            image = Image.open(str(image_path)).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            raise ValueError("Invalid image input type")
        
        # Prepare task
        task = {
            'image': image,
            'output_path': output_path,
            'restoration_type': restoration_type,
            **kwargs
        }
        
        # Execute restoration through orchestrator
        result = await self.orchestrator.process(task)
        
        return result
    
    async def execute_single_agent(self, 
                                 agent_name: str,
                                 image_path: Union[str, Path, Image.Image, np.ndarray],
                                 output_path: Optional[str] = None,
                                 **kwargs) -> Dict[str, Any]:
        """
        Execute a single specialized agent
        
        Args:
            agent_name: Name of the agent to execute
            image_path: Input image
            output_path: Output path
            **kwargs: Agent-specific parameters
            
        Returns:
            Dictionary containing agent results
        """
        if not self.is_initialized:
            await self.initialize()
        
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found. Available agents: {list(self.agents.keys())}")
        
        # Prepare input image
        if isinstance(image_path, (str, Path)):
            image = Image.open(str(image_path)).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            raise ValueError("Invalid image input type")
        
        # Prepare task
        task = {
            'image': image,
            'output_path': output_path,
            **kwargs
        }
        
        # Execute agent
        agent = self.agents[agent_name]
        result = await agent.process(task)
        
        return result
    
    async def execute_custom_pipeline(self,
                                    pipeline: List[str],
                                    image_path: Union[str, Path, Image.Image, np.ndarray],
                                    output_path: Optional[str] = None,
                                    **kwargs) -> Dict[str, Any]:
        """
        Execute a custom pipeline with specified agents
        
        Args:
            pipeline: List of agent names to execute in sequence
            image_path: Input image
            output_path: Output path
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing pipeline results
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Prepare input image
        if isinstance(image_path, (str, Path)):
            image = Image.open(str(image_path)).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            raise ValueError("Invalid image input type")
        
        # Prepare task
        task = {
            'image': image,
            'output_path': output_path,
            **kwargs
        }
        
        # Execute custom pipeline
        result = await self.orchestrator.execute_custom_pipeline(pipeline, task)
        
        return result
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent"""
        return self.orchestrator.get_agent_info(agent_name)
    
    async def forensic_analysis(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform forensic analysis on image"""
        return await self.execute_single_agent('forensic_analysis', image_path)
    
    async def material_recognition(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform material recognition analysis"""
        return await self.execute_single_agent('material_recognition', image_path)
    
    async def damage_classification(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Classify damage types in image"""
        return await self.execute_single_agent('damage_classifier', image_path)
    
    async def context_aware_restoration(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform context-aware restoration"""
        return await self.execute_single_agent('context_aware_restoration', image_path)
    
    async def adaptive_enhancement(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform adaptive enhancement"""
        return await self.execute_single_agent('adaptive_enhancement', image_path)
    
    async def meta_correction(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform meta-correction and self-critique"""
        return await self.execute_single_agent('meta_correction', image_path)
    
    async def self_critique(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform self-critique analysis"""
        return await self.execute_single_agent('self_critique', image_path)
    
    async def hyperspectral_recovery(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform hyperspectral texture recovery"""
        return await self.execute_single_agent('hyperspectral_recovery', image_path)
    
    async def paint_layer_decomposition(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform paint layer decomposition"""
        return await self.execute_single_agent('paint_layer_decomposition', image_path)
    
    async def scientific_restoration(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform scientific restoration pipeline"""
        scientific_pipeline = [
            'forensic_analysis',
            'material_recognition', 
            'damage_classifier',
            'hyperspectral_recovery',
            'paint_layer_decomposition',
            'context_aware_restoration',
            'meta_correction'
        ]
        return await self.execute_custom_pipeline(scientific_pipeline, image_path)
    
    async def artistic_restoration(self, image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """Perform artistic restoration pipeline"""
        artistic_pipeline = [
            'style_aesthetic',
            'semantic_editing',
            'generative',
            'adaptive_enhancement',
            'auto_retouch',
            'self_critique'
        ]
        return await self.execute_custom_pipeline(artistic_pipeline, image_path)
    
    async def cleanup(self):
        """Cleanup all resources"""
        try:
            await self.orchestrator.cleanup()
            for agent in self.agents.values():
                await agent.cleanup()
            logger.info("AISIS cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        asyncio.create_task(self.cleanup())

    async def edit_image(self, image_path, instruction, output_path=None, **kwargs):
        """
        Edit an image according to a natural language instruction using the semantic editing agent.
        Args:
            image_path: Path to the input image.
            instruction: Text instruction for the edit.
            output_path: Optional path to save the edited image.
            **kwargs: Additional parameters.
        Returns:
            Result dictionary from the semantic editing agent.
        """
        if not self.is_initialized:
            await self.initialize()
        agent = self.agents.get('semantic_editing')
        if agent is None:
            raise RuntimeError("SemanticEditingAgent is not available.")
        # Prepare input image
        if isinstance(image_path, (str, Path)):
            image = Image.open(str(image_path)).convert('RGB')
        elif isinstance(image_path, Image.Image):
            image = image_path
        elif isinstance(image_path, np.ndarray):
            image = Image.fromarray(image_path)
        else:
            raise ValueError("Invalid image input type")
        task = {
            'image': image,
            'description': instruction,
            'output_path': output_path,
            **kwargs
        }
        return await agent.process(task)

    def start_voice_mode(self):
        """
        Start real-time voice interaction mode using the VoiceManager.
        Returns:
            None
        """
        import asyncio
        from src.core.voice_manager import VoiceManager
        async def run_voice():
            vm = VoiceManager()
            await vm.initialize()
            print("Voice mode started. Speak into your microphone (not implemented: add UI loop here).")
            # TODO: Add real-time audio capture and command loop
        asyncio.run(run_voice())

    @staticmethod
    def gpu_status():
        """
        Report CUDA/GPU availability and device info.
        Returns:
            Dict with CUDA status and device name.
        """
        import torch
        available = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if available else 'CPU'
        return {'cuda_available': available, 'device': device}

# Convenience functions
async def restore_image(image_path: Union[str, Path, Image.Image, np.ndarray],
                       output_path: Optional[str] = None,
                       restoration_type: str = 'comprehensive',
                       **kwargs) -> Dict[str, Any]:
    """
    Convenience function for quick image restoration
    
    Args:
        image_path: Input image
        output_path: Output path
        restoration_type: Type of restoration
        **kwargs: Additional parameters
        
    Returns:
        Restoration results
    """
    async with AISIS() as aisis:
        return await aisis.restore_image(image_path, output_path, restoration_type, **kwargs)

async def forensic_analysis(image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Convenience function for forensic analysis"""
    async with AISIS() as aisis:
        return await aisis.forensic_analysis(image_path)

async def scientific_restoration(image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Convenience function for scientific restoration"""
    async with AISIS() as aisis:
        return await aisis.scientific_restoration(image_path)

async def artistic_restoration(image_path: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Any]:
    """Convenience function for artistic restoration"""
    async with AISIS() as aisis:
        return await aisis.artistic_restoration(image_path)

__version__ = "2.0.0"
__author__ = "AISIS Development Team"
__description__ = "Professional-grade AI image restoration and enhancement system"
