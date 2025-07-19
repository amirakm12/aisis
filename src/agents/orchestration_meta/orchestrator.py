"""
Orchestrator Agent
Coordinates and manages all specialized restoration agents
"""

import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

import json
import importlib
from pathlib import Path

from src.agents.base_agent import BaseAgent
from .image_restoration import ImageRestorationAgent
from .style_aesthetic import StyleAestheticAgent
from .semantic_editing import SemanticEditingAgent
from .auto_retouch import AutoRetouchAgent
from .generative import GenerativeAgent
from .neural_radiance import NeuralRadianceAgent
from .denoising import DenoisingAgent
from .super_resolution import SuperResolutionAgent
from .color_correction import ColorCorrectionAgent
from .tile_stitching import TileStitchingAgent
from .text_recovery import TextRecoveryAgent
from .feedback_loop import FeedbackLoopAgent
from .perspective_correction import PerspectiveCorrectionAgent
from .material_recognition import MaterialRecognitionAgent
from .damage_classifier import DamageClassifierAgent
from .hyperspectral_recovery import HyperspectralRecoveryAgent
from .paint_layer_decomposition import PaintLayerDecompositionAgent
from .meta_correction import MetaCorrectionAgent
from .self_critique import SelfCritiqueAgent
from .forensic_analysis import ForensicAnalysisAgent
from .context_aware_restoration import ContextAwareRestorationAgent
from .adaptive_enhancement import AdaptiveEnhancementAgent
from .hyper_orchestrator import HyperOrchestrator

class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("OrchestratorAgent")
        self.agents = {}
        self.restoration_pipeline = []
        
    async def _initialize(self) -> None:
        """Initialize all restoration agents dynamically from config"""
        try:
            logger.info("Initializing comprehensive restoration pipeline dynamically...")
            
            # Load config
            config_path = Path(__file__).parent.parent.parent.parent / 'config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.agents = {}
            for category, agent_names in config.get('agents', {}).items():
                for agent_name in agent_names:
                    module_path = f"src.agents.{category}.{agent_name}"
                    mod = importlib.import_module(module_path)
                    class_name = ''.join(word.capitalize() for word in agent_name.split('_')) + 'Agent'
                    agent_class = getattr(mod, class_name)
                    self.agents[agent_name] = agent_class()
            
            # Initialize all agents
            for name, agent in self.agents.items():
                logger.info(f"Initializing {name}...")
                await agent.initialize()
            
            # Define restoration pipeline stages
            self.restoration_pipeline = [
                'forensic_analysis',           # Scientific examination
                'material_recognition',        # Material identification
                'damage_classifier',           # Damage assessment
                'context_aware_restoration',   # Context-based restoration
                'image_restoration',           # Core restoration
                'denoising',                   # Noise removal
                'color_correction',            # Color restoration
                'perspective_correction',      # Geometric correction
                'super_resolution',            # Resolution enhancement
                'text_recovery',               # Text restoration
                'paint_layer_decomposition',   # Layer analysis
                'hyperspectral_recovery',      # Spectral restoration
                'semantic_editing',            # Content-aware editing
                'style_aesthetic',             # Style enhancement
                'adaptive_enhancement',        # Intelligent enhancement
                'auto_retouch',                # Automated retouching
                'generative',                  # Generative restoration
                'neural_radiance',             # 3D reconstruction
                'tile_stitching',              # Large image handling
                'feedback_loop',               # Quality feedback
                'self_critique',               # Self-assessment
                'meta_correction'              # Final corrections
            ]
            
            logger.info(f"Orchestrator initialized with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def _process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process restoration task through comprehensive pipeline"""
        try:
            input_image = task.get('image')
            restoration_type = task.get('restoration_type', 'comprehensive')
            output_path = task.get('output_path')
            
            logger.info(f"Starting {restoration_type} restoration pipeline")
            
            # Initialize result tracking
            results = {
                'input_image': input_image,
                'restoration_type': restoration_type,
                'pipeline_stages': [],
                'final_output': None,
                'quality_metrics': {},
                'agent_reports': {}
            }
            
            current_image = input_image
            
            # Execute restoration pipeline
            for stage_name in self.restoration_pipeline:
                if stage_name in self.agents:
                    agent = self.agents[stage_name]
                    
                    logger.info(f"Executing {stage_name} stage...")
                    
                    # Prepare stage task
                    stage_task = {
                        'image': current_image,
                        'output_path': None,  # Don't save intermediate results
                        'restoration_type': restoration_type,
                        'previous_results': results
                    }
                    
                    # Execute stage
                    stage_result = await agent.process(stage_task)
                    
                    # Update current image if stage produced output
                    if 'output_image' in stage_result:
                        current_image = stage_result['output_image']
                    
                    # Record stage results
                    results['pipeline_stages'].append({
                        'stage': stage_name,
                        'status': stage_result.get('status', 'unknown'),
                        'output_image': stage_result.get('output_image'),
                        'metrics': stage_result.get('quality_metrics', {}),
                        'analysis': stage_result.get('analysis', {})
                    })
                    
                    # Store agent-specific reports
                    results['agent_reports'][stage_name] = stage_result
                    
                    logger.info(f"Completed {stage_name} stage")
            
            # Save final result
            if output_path and current_image:
                current_image.save(output_path)
                results['final_output'] = output_path
            
            # Calculate overall quality metrics
            results['quality_metrics'] = self._calculate_overall_quality(results)
            
            logger.info("Restoration pipeline completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"Orchestrator processing failed: {e}")
            raise
    
    def _calculate_overall_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics from pipeline results"""
        quality_metrics = {
            'overall_score': 0.0,
            'stage_scores': {},
            'improvements': {},
            'recommendations': []
        }
        
        total_score = 0.0
        stage_count = 0
        
        for stage in results['pipeline_stages']:
            stage_name = stage['stage']
            stage_metrics = stage.get('metrics', {})
            
            # Calculate stage score
            stage_score = 0.0
            if 'quality_score' in stage_metrics:
                stage_score = stage_metrics['quality_score']
            elif 'improvement_score' in stage_metrics:
                stage_score = stage_metrics['improvement_score']
            elif 'confidence_score' in stage_metrics:
                stage_score = stage_metrics['confidence_score']
            
            quality_metrics['stage_scores'][stage_name] = stage_score
            total_score += stage_score
            stage_count += 1
            
            # Record improvements
            if 'improvements' in stage_metrics:
                quality_metrics['improvements'][stage_name] = stage_metrics['improvements']
            
            # Collect recommendations
            if 'recommendations' in stage_metrics:
                quality_metrics['recommendations'].extend(stage_metrics['recommendations'])
        
        # Calculate overall score
        if stage_count > 0:
            quality_metrics['overall_score'] = total_score / stage_count
        
        return quality_metrics
    
    async def execute_single_agent(self, agent_name: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        return await agent.process(task)
    
    async def execute_custom_pipeline(self, pipeline: List[str], task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a custom pipeline with specified agents"""
        results = {
            'input_image': task.get('image'),
            'custom_pipeline': pipeline,
            'pipeline_stages': [],
            'final_output': None
        }
        
        current_image = task.get('image')
        
        for stage_name in pipeline:
            if stage_name in self.agents:
                agent = self.agents[stage_name]
                
                stage_task = {
                    'image': current_image,
                    'output_path': None,
                    'restoration_type': 'custom'
                }
                
                stage_result = await agent.process(stage_task)
                
                if 'output_image' in stage_result:
                    current_image = stage_result['output_image']
                
                results['pipeline_stages'].append({
                    'stage': stage_name,
                    'status': stage_result.get('status', 'unknown'),
                    'output_image': stage_result.get('output_image')
                })
        
        if current_image:
            results['final_output'] = current_image
        
        return results
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents"""
        return list(self.agents.keys())
    
    def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent"""
        if agent_name not in self.agents:
            return {'error': f'Agent {agent_name} not found'}
        
        agent = self.agents[agent_name]
        return {
            'name': agent_name,
            'class': agent.__class__.__name__,
            'description': agent.__doc__ or 'No description available',
            'capabilities': getattr(agent, 'capabilities', []),
            'status': 'initialized' if agent.is_initialized else 'not_initialized'
        }
    
    async def _cleanup(self) -> None:
        """Cleanup all agents"""
        for name, agent in self.agents.items():
            try:
                await agent.cleanup()
                logger.info(f"Cleaned up {name} agent")
            except Exception as e:
                logger.error(f"Failed to cleanup {name} agent: {e}")
        
        self.agents.clear()
