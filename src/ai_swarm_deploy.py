#!/usr/bin/env python3
"""
AI Agent Swarm Deployment Script
Deploys a distributed, cooperative swarm of specialized AI agents for extreme performance optimization
"""

import asyncio
import signal
import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from ai_swarm.core.orchestrator import SwarmOrchestrator, SwarmConfiguration
from ai_swarm.core.communication import MessagePriority


class SwarmDeployer:
    """Main deployment manager for the AI agent swarm"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.orchestrator: Optional[SwarmOrchestrator] = None
        self.config = self._load_configuration(config_file)
        self.deployment_stats = {
            "start_time": None,
            "deployment_duration": 0.0,
            "peak_agents": 0,
            "total_tasks_processed": 0
        }
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger("SwarmDeployer")
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _load_configuration(self, config_file: Optional[str]) -> SwarmConfiguration:
        """Load configuration from file or use defaults"""
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                
                return SwarmConfiguration(
                    max_agents=config_data.get("max_agents", 32),
                    communication_channel_size=config_data.get("channel_size", 1024 * 1024),
                    monitoring_interval=config_data.get("monitoring_interval", 1.0),
                    optimization_interval=config_data.get("optimization_interval", 5.0),
                    fault_tolerance_enabled=config_data.get("fault_tolerance", True),
                    load_balancing_enabled=config_data.get("load_balancing", True),
                    auto_scaling_enabled=config_data.get("auto_scaling", True),
                    telemetry_retention_hours=config_data.get("telemetry_retention", 24)
                )
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
                print("Using default configuration...")
        
        return SwarmConfiguration()
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        # Create logs directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"swarm_{int(time.time())}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set specific log levels for different components
        logging.getLogger("SwarmOrchestrator").setLevel(logging.INFO)
        logging.getLogger("SwarmCommunicator").setLevel(logging.INFO)
        logging.getLogger("Agent").setLevel(logging.INFO)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            if self.orchestrator:
                asyncio.create_task(self.orchestrator.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def deploy(self):
        """Deploy the AI agent swarm"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("🚀 DEPLOYING DISTRIBUTED AI AGENT SWARM")
            self.logger.info("=" * 80)
            
            self.deployment_stats["start_time"] = time.time()
            
            # Display system information
            await self._display_system_info()
            
            # Create and configure orchestrator
            self.logger.info("Initializing swarm orchestrator...")
            self.orchestrator = SwarmOrchestrator(self.config)
            
            # Add event handlers
            self._setup_event_handlers()
            
            # Start the swarm
            self.logger.info("Starting AI agent swarm deployment...")
            await self.orchestrator.start()
            
        except KeyboardInterrupt:
            self.logger.info("Deployment interrupted by user")
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
        finally:
            if self.orchestrator:
                await self.orchestrator.shutdown()
            
            # Display final statistics
            await self._display_final_stats()
    
    async def _display_system_info(self):
        """Display system information and capabilities"""
        import psutil
        import platform
        
        self.logger.info("System Information:")
        self.logger.info(f"  Platform: {platform.system()} {platform.release()}")
        self.logger.info(f"  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
        self.logger.info(f"  Memory: {psutil.virtual_memory().total // (1024**3)} GB")
        self.logger.info(f"  Python: {platform.python_version()}")
        
        # Display configuration
        self.logger.info("Swarm Configuration:")
        self.logger.info(f"  Max Agents: {self.config.max_agents}")
        self.logger.info(f"  Channel Size: {self.config.communication_channel_size // 1024} KB")
        self.logger.info(f"  Monitoring Interval: {self.config.monitoring_interval}s")
        self.logger.info(f"  Fault Tolerance: {self.config.fault_tolerance_enabled}")
        self.logger.info(f"  Auto Scaling: {self.config.auto_scaling_enabled}")
        self.logger.info(f"  Load Balancing: {self.config.load_balancing_enabled}")
    
    def _setup_event_handlers(self):
        """Setup event handlers for monitoring"""
        async def agent_created_handler(event_data):
            agent_id = event_data["agent_id"]
            agent_type = event_data["agent_type"]
            self.logger.info(f"✅ Agent Created: {agent_type} -> {agent_id}")
            
            # Update peak agents
            if self.orchestrator:
                current_agents = len(self.orchestrator.agents)
                if current_agents > self.deployment_stats["peak_agents"]:
                    self.deployment_stats["peak_agents"] = current_agents
        
        async def agent_destroyed_handler(event_data):
            agent_id = event_data["agent_id"]
            self.logger.info(f"🗑️  Agent Destroyed: {agent_id}")
        
        if self.orchestrator:
            self.orchestrator.add_event_handler("agent_created", agent_created_handler)
            self.orchestrator.add_event_handler("agent_destroyed", agent_destroyed_handler)
    
    async def _display_final_stats(self):
        """Display final deployment statistics"""
        if self.deployment_stats["start_time"]:
            self.deployment_stats["deployment_duration"] = time.time() - self.deployment_stats["start_time"]
        
        if self.orchestrator:
            system_status = self.orchestrator.get_system_status()
            self.deployment_stats["total_tasks_processed"] = system_status["swarm_metrics"]["total_tasks_completed"]
        
        self.logger.info("=" * 80)
        self.logger.info("📊 DEPLOYMENT STATISTICS")
        self.logger.info("=" * 80)
        self.logger.info(f"Deployment Duration: {self.deployment_stats['deployment_duration']:.2f} seconds")
        self.logger.info(f"Peak Agents: {self.deployment_stats['peak_agents']}")
        self.logger.info(f"Total Tasks Processed: {self.deployment_stats['total_tasks_processed']}")
        
        if self.orchestrator:
            system_status = self.orchestrator.get_system_status()
            swarm_metrics = system_status["swarm_metrics"]
            
            self.logger.info(f"Final Active Agents: {swarm_metrics['active_agents']}")
            self.logger.info(f"System Throughput: {swarm_metrics['system_throughput']:.2f} tasks/sec")
            self.logger.info(f"Average Response Time: {swarm_metrics['average_response_time']:.4f}s")
            self.logger.info(f"Communication Latency: {swarm_metrics['communication_latency']:.6f}s")
            
            orchestrator_stats = system_status["orchestrator_stats"]
            self.logger.info(f"Agents Created: {orchestrator_stats['agents_created']}")
            self.logger.info(f"Agents Destroyed: {orchestrator_stats['agents_destroyed']}")
            self.logger.info(f"Failovers Executed: {orchestrator_stats['failovers_executed']}")
            self.logger.info(f"Optimizations Performed: {orchestrator_stats['optimizations_performed']}")
        
        self.logger.info("=" * 80)
        self.logger.info("🏁 AI AGENT SWARM DEPLOYMENT COMPLETE")
        self.logger.info("=" * 80)


async def create_sample_workloads(orchestrator: SwarmOrchestrator):
    """Create sample workloads to demonstrate the system"""
    import numpy as np
    import uuid
    
    logger = logging.getLogger("SampleWorkloads")
    logger.info("Creating sample workloads to demonstrate system capabilities...")
    
    # Wait for system to stabilize
    await asyncio.sleep(5)
    
    sample_workloads = [
        {
            "workload_id": f"matrix_multiply_{uuid.uuid4().hex[:8]}",
            "workload_type": "matrix_multiply",
            "data": [np.random.random((100, 100)).astype(np.float32), 
                    np.random.random((100, 100)).astype(np.float32)],
            "priority": 7,
            "target_latency": 0.01
        },
        {
            "workload_id": f"element_wise_{uuid.uuid4().hex[:8]}",
            "workload_type": "element_wise_add",
            "data": [np.random.random(10000).astype(np.float32),
                    np.random.random(10000).astype(np.float32)],
            "priority": 5,
            "target_latency": 0.001
        },
        {
            "workload_id": f"reduction_{uuid.uuid4().hex[:8]}",
            "workload_type": "reduction",
            "data": np.random.random(100000).astype(np.float32),
            "priority": 6,
            "target_latency": 0.005
        }
    ]
    
    # Send workloads to compute agents
    for workload in sample_workloads:
        # Find a compute agent
        compute_agents = [aid for aid in orchestrator.agents.keys() 
                         if aid.startswith("compute_agent")]
        
        if compute_agents:
            target_agent = compute_agents[0]  # Use first available compute agent
            
            try:
                message = {
                    "sender_id": "orchestrator",
                    "recipient_id": target_agent,
                    "message_type": "execute_workload",
                    "payload": workload,
                    "priority": MessagePriority.NORMAL
                }
                
                await orchestrator.communicator.send_message(message)
                logger.info(f"Sent workload {workload['workload_id']} to {target_agent}")
                
            except Exception as e:
                logger.error(f"Failed to send workload {workload['workload_id']}: {e}")
        
        # Small delay between workloads
        await asyncio.sleep(1)
    
    logger.info("Sample workloads created and dispatched")


async def monitor_system_performance(orchestrator: SwarmOrchestrator, duration: int = 60):
    """Monitor system performance for a specified duration"""
    logger = logging.getLogger("PerformanceMonitor")
    logger.info(f"Starting performance monitoring for {duration} seconds...")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            status = orchestrator.get_system_status()
            swarm_metrics = status["swarm_metrics"]
            
            logger.info(
                f"📈 Performance: "
                f"Agents: {swarm_metrics['active_agents']}/{swarm_metrics['total_agents']}, "
                f"Throughput: {swarm_metrics['system_throughput']:.2f} tasks/sec, "
                f"Latency: {swarm_metrics['average_response_time']:.4f}s, "
                f"Comm: {swarm_metrics['communication_latency']:.6f}s"
            )
            
            # Display agent-specific stats
            for agent_id, agent in orchestrator.agents.items():
                if hasattr(agent, 'get_agent_statistics'):
                    stats = agent.get_agent_statistics()
                    logger.debug(f"Agent {agent_id}: {stats}")
            
            await asyncio.sleep(10)  # Monitor every 10 seconds
            
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            await asyncio.sleep(5)
    
    logger.info("Performance monitoring completed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Deploy distributed AI agent swarm for extreme performance optimization"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Configuration file path (JSON format)"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run demonstration workloads"
    )
    parser.add_argument(
        "--monitor-duration", "-m",
        type=int,
        default=300,
        help="Performance monitoring duration in seconds (default: 300)"
    )
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    async def run_deployment():
        """Run the deployment with optional demo and monitoring"""
        deployer = SwarmDeployer(args.config)
        
        try:
            # Start deployment in background
            deployment_task = asyncio.create_task(deployer.deploy())
            
            # Wait for system to start
            await asyncio.sleep(10)
            
            if args.demo and deployer.orchestrator:
                # Create sample workloads
                workload_task = asyncio.create_task(
                    create_sample_workloads(deployer.orchestrator)
                )
                
                # Monitor performance
                monitor_task = asyncio.create_task(
                    monitor_system_performance(
                        deployer.orchestrator, 
                        args.monitor_duration
                    )
                )
                
                # Wait for demo tasks to complete
                await asyncio.gather(workload_task, monitor_task, return_exceptions=True)
            
            # Wait for deployment to complete or be interrupted
            await deployment_task
            
        except KeyboardInterrupt:
            print("\n🛑 Deployment interrupted by user")
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return 1
        
        return 0
    
    # Run the deployment
    try:
        exit_code = asyncio.run(run_deployment())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
        sys.exit(130)  # Standard exit code for Ctrl+C


if __name__ == "__main__":
    main()