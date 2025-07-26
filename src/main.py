#!/usr/bin/env python3
"""
AISIS Creative Studio - Main Application Launcher
Advanced AI system with comprehensive error recovery, memory management, and agent orchestration
"""

import asyncio
import sys
import argparse
import signal
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from loguru import logger

# Core system imports
from core.memory_manager import setup_memory_management, memory_manager
from core.model_manager import setup_model_management, model_manager
from core.error_recovery import setup_error_recovery, error_recovery
from core.config_validator import setup_config_validation, config_validator, ValidationLevel

# Agent imports
from agents.base_agent import BaseAgent, AgentState
from agents.image_restoration_agent import ImageRestorationAgent


class AISISApplication:
    """
    Main AISIS Creative Studio Application
    Manages the complete AI system lifecycle with advanced features
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize AISIS application
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        self.initialized = False
        
        # System components
        self.memory_manager = None
        self.model_manager = None
        self.error_recovery = None
        self.config_validator = None
        
        # Agents
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_registry = {}
        
        # Application state
        self.running = False
        self.shutdown_requested = False
        
        # Performance monitoring
        self.startup_time = None
        self.task_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0
        }
        
        logger.info("AISIS Application created")
    
    async def initialize(self) -> bool:
        """
        Initialize the complete AISIS system
        
        Returns:
            Success status
        """
        try:
            logger.info("🚀 Initializing AISIS Creative Studio...")
            
            # Load and validate configuration
            await self._load_configuration()
            
            # Initialize core systems
            await self._initialize_core_systems()
            
            # Initialize agents
            await self._initialize_agents()
            
            # Setup monitoring and callbacks
            self._setup_monitoring()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.initialized = True
            self.startup_time = asyncio.get_event_loop().time()
            
            logger.info("✅ AISIS Creative Studio initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AISIS: {e}")
            await self._emergency_cleanup()
            return False
    
    async def _load_configuration(self):
        """Load and validate application configuration"""
        logger.info("Loading configuration...")
        
        # Default configuration
        default_config = {
            "app": {
                "name": "AISIS Creative Studio",
                "version": "1.0.0",
                "debug": False
            },
            "memory": {
                "max_usage_gb": 16.0,
                "monitoring_enabled": True,
                "cleanup_threshold": 0.85
            },
            "models": {
                "cache_dir": "./models",
                "max_concurrent_downloads": 2,
                "auto_download": True
            },
            "agents": {
                "max_concurrent_tasks": 4,
                "task_timeout": 300.0,
                "auto_initialize": True
            },
            "recovery": {
                "state_dir": "./recovery_state",
                "checkpoint_interval": 60.0,
                "max_retries": 3
            }
        }
        
        # Load from file if provided
        if self.config_path:
            config_file = Path(self.config_path)
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        if config_file.suffix.lower() == '.json':
                            file_config = json.load(f)
                        else:
                            import yaml
                            file_config = yaml.safe_load(f)
                    
                    # Merge configurations
                    self.config = {**default_config, **file_config}
                    logger.info(f"Configuration loaded from {config_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to load config file: {e}")
                    self.config = default_config
            else:
                logger.warning(f"Config file not found: {config_file}, using defaults")
                self.config = default_config
        else:
            self.config = default_config
        
        # Validate configuration
        self.config_validator = setup_config_validation(ValidationLevel.STANDARD)
        validation_result = self.config_validator.validate_config(self.config)
        
        if not validation_result.valid:
            logger.error("Configuration validation failed:")
            for error in validation_result.errors:
                logger.error(f"  - {error}")
            raise ValueError("Invalid configuration")
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Config warning: {warning}")
        
        if validation_result.security_issues:
            for issue in validation_result.security_issues:
                logger.warning(f"Security issue: {issue}")
        
        # Use sanitized config
        self.config = validation_result.sanitized_config
        logger.info("Configuration validated successfully")
    
    async def _initialize_core_systems(self):
        """Initialize core system components"""
        logger.info("Initializing core systems...")
        
        # Memory management
        self.memory_manager = setup_memory_management(auto_start=True)
        logger.info("✅ Memory management initialized")
        
        # Model management
        models_dir = self.config.get("models", {}).get("cache_dir", "./models")
        self.model_manager = setup_model_management(models_dir)
        logger.info("✅ Model management initialized")
        
        # Error recovery
        state_dir = self.config.get("recovery", {}).get("state_dir", "./recovery_state")
        self.error_recovery = setup_error_recovery(state_dir)
        logger.info("✅ Error recovery initialized")
        
        # Register default models
        await self._register_default_models()
        
        logger.info("Core systems initialized successfully")
    
    async def _register_default_models(self):
        """Register default AI models"""
        logger.info("Registering default models...")
        
        default_models = [
            ("whisper-base", "openai/whisper-base", 0.6),
            ("clip-vit-base-patch32", "openai/clip-vit-base-patch32", 0.6),
            ("bert-base-uncased", "bert-base-uncased", 0.4),
        ]
        
        for name, model_id, size_gb in default_models:
            try:
                self.model_manager.register_model(name, model_id, size_gb)
                logger.debug(f"Registered model: {name}")
            except Exception as e:
                logger.error(f"Failed to register model {name}: {e}")
        
        logger.info(f"Registered {len(default_models)} default models")
    
    async def _initialize_agents(self):
        """Initialize AI agents"""
        logger.info("Initializing agents...")
        
        # Register agent types
        self.agent_registry = {
            "image_restoration": ImageRestorationAgent,
            # Add more agent types here
        }
        
        # Initialize agents based on configuration
        agent_config = self.config.get("agents", {})
        auto_initialize = agent_config.get("auto_initialize", True)
        
        if auto_initialize:
            # Initialize default agents
            for agent_name, agent_class in self.agent_registry.items():
                try:
                    agent = agent_class()
                    success = await agent.initialize()
                    
                    if success:
                        self.agents[agent_name] = agent
                        logger.info(f"✅ Agent initialized: {agent_name}")
                    else:
                        logger.error(f"❌ Failed to initialize agent: {agent_name}")
                        
                except Exception as e:
                    logger.error(f"Error initializing agent {agent_name}: {e}")
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    def _setup_monitoring(self):
        """Setup system monitoring and callbacks"""
        logger.info("Setting up monitoring...")
        
        # Memory monitoring callbacks
        def memory_callback():
            stats = self.memory_manager.get_memory_stats()
            if stats.ram_percent > 0.9:
                logger.critical(f"Critical memory usage: {stats.ram_percent*100:.1f}%")
        
        self.memory_manager.add_cleanup_callback(memory_callback)
        
        # Error recovery callbacks
        async def error_callback(error_info, recovery_action, success):
            self.task_stats["failed"] += 1
            logger.warning(f"Error handled: {error_info.error_type} -> {recovery_action.value}")
        
        self.error_recovery.add_recovery_callback(error_callback)
        
        # Agent performance callbacks
        def performance_callback(agent, task, success):
            self.task_stats["total_processed"] += 1
            if success:
                self.task_stats["successful"] += 1
            else:
                self.task_stats["failed"] += 1
        
        for agent in self.agents.values():
            agent.add_performance_callback(performance_callback)
        
        logger.info("Monitoring setup completed")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_requested = True
            asyncio.create_task(self.shutdown())
        
        # Setup signal handlers
        for sig in [signal.SIGTERM, signal.SIGINT]:
            try:
                signal.signal(sig, signal_handler)
            except (OSError, ValueError):
                # Signal handling might not be available in all environments
                pass
    
    async def run(self):
        """Run the main application loop"""
        if not self.initialized:
            logger.error("Application not initialized")
            return False
        
        logger.info("🎯 Starting AISIS Creative Studio...")
        self.running = True
        
        try:
            # Main application loop
            while self.running and not self.shutdown_requested:
                # Process any pending tasks
                await self._process_system_tasks()
                
                # Check system health
                await self._health_check()
                
                # Brief pause
                await asyncio.sleep(1.0)
            
            logger.info("Main loop exited")
            return True
            
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            await self.error_recovery.handle_error(e, {"component": "main_loop"})
            return False
    
    async def _process_system_tasks(self):
        """Process system-level tasks"""
        # This would handle system-level task coordination
        # For now, just check agent status
        
        for agent_name, agent in self.agents.items():
            if agent.state == AgentState.ERROR:
                logger.warning(f"Agent {agent_name} in error state, attempting restart")
                try:
                    await agent.initialize()
                except Exception as e:
                    logger.error(f"Failed to restart agent {agent_name}: {e}")
    
    async def _health_check(self):
        """Perform system health check"""
        # Check memory usage
        memory_stats = self.memory_manager.get_memory_stats()
        if memory_stats.ram_percent > 0.95:
            logger.critical("Critical memory usage detected")
            self.memory_manager.cleanup_memory(force=True)
        
        # Check agent health
        unhealthy_agents = [name for name, agent in self.agents.items() 
                          if agent.state in [AgentState.ERROR, AgentState.SHUTDOWN]]
        
        if unhealthy_agents:
            logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
    
    async def submit_task(self, 
                         agent_name: str, 
                         task_type: str, 
                         data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a task to a specific agent
        
        Args:
            agent_name: Name of the agent
            task_type: Type of task
            data: Task data
            
        Returns:
            Task result
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        if not agent.is_ready:
            raise RuntimeError(f"Agent {agent_name} is not ready")
        
        try:
            task = await agent.submit_task(task_type, data)
            logger.info(f"Task submitted to {agent_name}: {task.task_id}")
            
            # Wait for task completion (simplified)
            while task.task_id in agent.active_tasks:
                await asyncio.sleep(0.1)
            
            # Find completed task
            completed_task = None
            for completed in agent.completed_tasks:
                if completed.task_id == task.task_id:
                    completed_task = completed
                    break
            
            if completed_task:
                return completed_task.result or {"status": "error", "error": completed_task.error}
            else:
                return {"status": "error", "error": "Task not found in completed tasks"}
                
        except Exception as e:
            logger.error(f"Task submission failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = asyncio.get_event_loop().time() - self.startup_time if self.startup_time else 0
        
        return {
            "application": {
                "name": self.config.get("app", {}).get("name", "AISIS"),
                "version": self.config.get("app", {}).get("version", "1.0.0"),
                "initialized": self.initialized,
                "running": self.running,
                "uptime_seconds": uptime
            },
            "memory": self.memory_manager.get_memory_stats().__dict__ if self.memory_manager else {},
            "agents": {
                name: agent.get_status() for name, agent in self.agents.items()
            },
            "models": {
                "registered": len(self.model_manager.list_models()) if self.model_manager else 0,
                "loaded": len(self.model_manager.get_loaded_models()) if self.model_manager else 0
            },
            "tasks": self.task_stats,
            "errors": self.error_recovery.get_error_statistics() if self.error_recovery else {}
        }
    
    async def shutdown(self):
        """Graceful application shutdown"""
        if not self.running:
            return
        
        logger.info("🛑 Shutting down AISIS Creative Studio...")
        self.running = False
        
        try:
            # Shutdown agents
            for agent_name, agent in self.agents.items():
                try:
                    logger.info(f"Shutting down agent: {agent_name}")
                    await agent.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down agent {agent_name}: {e}")
            
            # Cleanup core systems
            if self.model_manager:
                await self.model_manager.cleanup()
            
            if self.memory_manager:
                self.memory_manager.stop_monitoring()
            
            if self.error_recovery:
                self.error_recovery.stop_checkpointing()
            
            logger.info("✅ AISIS Creative Studio shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def _emergency_cleanup(self):
        """Emergency cleanup in case of initialization failure"""
        logger.critical("Performing emergency cleanup...")
        
        try:
            if self.memory_manager:
                self.memory_manager.cleanup_memory(force=True)
                self.memory_manager.stop_monitoring()
            
            if self.error_recovery:
                self.error_recovery.stop_checkpointing()
            
        except Exception as e:
            logger.error(f"Emergency cleanup error: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AISIS Creative Studio")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    parser.add_argument("--status", "-s", action="store_true", help="Show system status and exit")
    parser.add_argument("--test-task", help="Submit a test task (format: agent:task_type:image_path)")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG", format="{time} | {level} | {name}:{function}:{line} | {message}")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    
    # Create and initialize application
    app = AISISApplication(config_path=args.config)
    
    try:
        # Initialize
        success = await app.initialize()
        if not success:
            logger.error("Failed to initialize application")
            sys.exit(1)
        
        if args.status:
            # Show status and exit
            status = app.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            return
        
        if args.test_task:
            # Submit test task
            try:
                parts = args.test_task.split(":")
                if len(parts) != 3:
                    raise ValueError("Test task format: agent:task_type:image_path")
                
                agent_name, task_type, image_path = parts
                
                result = await app.submit_task(agent_name, task_type, {"image_path": image_path})
                print(f"Task result: {json.dumps(result, indent=2, default=str)}")
                
            except Exception as e:
                logger.error(f"Test task failed: {e}")
                sys.exit(1)
        else:
            # Run main application
            await app.run()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())