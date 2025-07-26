#!/usr/bin/env python3
"""
AI Agent Swarm Demonstration
Shows the distributed AI agent swarm system in action
"""

import asyncio
import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ai_swarm.core.orchestrator import SwarmOrchestrator, SwarmConfiguration
from ai_swarm.core.communication import MessagePriority


async def demo_swarm_deployment():
    """Demonstrate the AI agent swarm system"""
    print("🚀 AI AGENT SWARM DEMONSTRATION")
    print("=" * 50)
    
    # Create configuration
    config = SwarmConfiguration(
        max_agents=8,
        monitoring_interval=2.0,
        optimization_interval=10.0,
        fault_tolerance_enabled=True,
        auto_scaling_enabled=True,
        load_balancing_enabled=True
    )
    
    # Create orchestrator
    print("🔧 Initializing swarm orchestrator...")
    orchestrator = SwarmOrchestrator(config)
    
    try:
        # Start the swarm (this will create initial agents)
        print("⚡ Starting AI agent swarm...")
        
        # Start orchestrator in background
        orchestrator_task = asyncio.create_task(orchestrator.start())
        
        # Wait for system to initialize
        await asyncio.sleep(5)
        
        print(f"✅ Swarm started with {len(orchestrator.agents)} agents")
        
        # Display agent information
        print("\n📋 Active Agents:")
        for agent_id, agent in orchestrator.agents.items():
            print(f"  - {agent_id}: {agent.__class__.__name__} (State: {agent.state.value})")
        
        # Create sample workloads
        print("\n🎯 Creating sample workloads...")
        await create_demo_workloads(orchestrator)
        
        # Monitor system performance
        print("\n📊 Monitoring system performance...")
        await monitor_performance(orchestrator, duration=30)
        
        # Display final statistics
        print("\n📈 Final System Statistics:")
        status = orchestrator.get_system_status()
        swarm_metrics = status["swarm_metrics"]
        
        print(f"  Total Agents: {swarm_metrics['total_agents']}")
        print(f"  Active Agents: {swarm_metrics['active_agents']}")
        print(f"  Tasks Completed: {swarm_metrics['total_tasks_completed']}")
        print(f"  System Throughput: {swarm_metrics['system_throughput']:.2f} tasks/sec")
        print(f"  Average Response Time: {swarm_metrics['average_response_time']:.4f}s")
        print(f"  Communication Latency: {swarm_metrics['communication_latency']:.6f}s")
        
        print("\n🏁 Demonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
    finally:
        # Shutdown the swarm
        print("\n🛑 Shutting down AI agent swarm...")
        await orchestrator.shutdown()
        print("✅ Swarm shutdown complete")


async def create_demo_workloads(orchestrator):
    """Create demonstration workloads"""
    workloads = [
        {
            "workload_id": "demo_matrix_multiply",
            "workload_type": "matrix_multiply", 
            "data": [np.random.random((50, 50)).astype(np.float32),
                    np.random.random((50, 50)).astype(np.float32)],
            "priority": 7,
            "target_latency": 0.01
        },
        {
            "workload_id": "demo_element_wise",
            "workload_type": "element_wise_add",
            "data": [np.random.random(5000).astype(np.float32),
                    np.random.random(5000).astype(np.float32)],
            "priority": 5,
            "target_latency": 0.001
        },
        {
            "workload_id": "demo_reduction",
            "workload_type": "reduction",
            "data": np.random.random(50000).astype(np.float32),
            "priority": 6,
            "target_latency": 0.005
        }
    ]
    
    # Find compute agents
    compute_agents = [aid for aid in orchestrator.agents.keys() 
                     if aid.startswith("compute_agent")]
    
    if not compute_agents:
        print("⚠️  No compute agents available for workloads")
        return
    
    # Send workloads
    for i, workload in enumerate(workloads):
        target_agent = compute_agents[i % len(compute_agents)]
        
        try:
            # Create message
            message = {
                "sender_id": "demo",
                "recipient_id": target_agent,
                "message_type": "execute_workload",
                "payload": workload,
                "priority": MessagePriority.NORMAL
            }
            
            # Send message
            await orchestrator.communicator.send_message(message)
            print(f"  📤 Sent {workload['workload_id']} to {target_agent}")
            
        except Exception as e:
            print(f"  ❌ Failed to send workload {workload['workload_id']}: {e}")
        
        # Small delay between workloads
        await asyncio.sleep(1)


async def monitor_performance(orchestrator, duration=30):
    """Monitor system performance"""
    print(f"  Monitoring for {duration} seconds...")
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            status = orchestrator.get_system_status()
            swarm_metrics = status["swarm_metrics"]
            
            print(f"  📊 Agents: {swarm_metrics['active_agents']}/{swarm_metrics['total_agents']}, "
                  f"Throughput: {swarm_metrics['system_throughput']:.2f} tasks/sec, "
                  f"Latency: {swarm_metrics['average_response_time']:.4f}s")
            
            await asyncio.sleep(5)
            
        except Exception as e:
            print(f"  ❌ Monitoring error: {e}")
            break
    
    print("  ✅ Performance monitoring completed")


def main():
    """Main entry point"""
    try:
        asyncio.run(demo_swarm_deployment())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())