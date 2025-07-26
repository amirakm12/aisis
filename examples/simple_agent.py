#!/usr/bin/env python3
"""
Simple example agent implementation.
Demonstrates how to create and use agents with the AISIS framework.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aisis.agents.base_agent import BaseAgent, AgentConfig, AgentResponse
from aisis.core.memory_manager import memory_manager
from aisis.core.error_handler import error_handler
import structlog

logger = structlog.get_logger(__name__)


class SimpleConversationalAgent(BaseAgent):
    """Simple conversational agent using a pre-trained model."""
    
    async def _generate_response(self, input_text: str, **kwargs) -> str:
        """Generate response using the loaded model."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(input_text + self.tokenizer.eos_token, return_tensors='pt')
            
            # Generate response
            with self.tokenizer.no_grad() if hasattr(self.tokenizer, 'no_grad') else contextlib.nullcontext():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the input part from response
            if response.startswith(input_text):
                response = response[len(input_text):].strip()
            
            return response if response else "I'm not sure how to respond to that."
            
        except Exception as e:
            self.logger.error("Response generation failed", error=str(e))
            return "I apologize, but I encountered an error while processing your request."


async def main():
    """Main demonstration function."""
    print("🤖 AISIS Simple Agent Example")
    print("=" * 40)
    
    # Start memory management
    memory_manager.start()
    
    try:
        # Create agent configuration
        agent_config = AgentConfig(
            name="simple_chat_agent",
            model_name="microsoft/DialoGPT-small",  # Using small model for demo
            max_tokens=100,
            temperature=0.7,
            memory_limit_gb=1.0,
            system_prompt="You are a helpful AI assistant."
        )
        
        print(f"📝 Creating agent: {agent_config.name}")
        print(f"🧠 Model: {agent_config.model_name}")
        
        # Create agent
        agent = SimpleConversationalAgent(agent_config)
        
        # Initialize agent (this will download and load the model)
        print("\n🔄 Initializing agent (this may take a while for first run)...")
        success = await agent.initialize()
        
        if not success:
            print("❌ Failed to initialize agent")
            return
        
        print("✅ Agent initialized successfully!")
        
        # Check agent status
        status = agent.get_status()
        print(f"\n📊 Agent Status:")
        print(f"  • Initialized: {status['is_initialized']}")
        print(f"  • Model Loaded: {status['is_model_loaded']}")
        print(f"  • Memory Usage: {status['memory_usage_gb']:.1f}GB")
        
        # Interactive chat loop
        print("\n💬 Starting interactive chat (type 'quit' to exit)")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Thinking...")
                
                # Process input
                response = await agent.process(user_input)
                
                if response.error:
                    print(f"❌ Error: {response.error}")
                else:
                    print(f"Bot: {response.content}")
                    print(f"⏱️  Processing time: {response.processing_time_ms}ms")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Unexpected error: {str(e)}")
        
        print("\n👋 Goodbye!")
        
        # Show final statistics
        final_status = agent.get_status()
        print(f"\n📈 Final Statistics:")
        print(f"  • Total Requests: {final_status['total_requests']}")
        print(f"  • Error Count: {final_status['error_count']}")
        print(f"  • Error Rate: {final_status['error_rate']:.1%}")
        
        # Health check
        health = await agent.health_check()
        print(f"  • Health Status: {'✅ Healthy' if health['healthy'] else '⚠️ Issues Detected'}")
        if health['issues']:
            for issue in health['issues']:
                print(f"    - {issue}")
        
    except Exception as e:
        logger.error("Example failed", error=str(e))
        print(f"❌ Example failed: {str(e)}")
    
    finally:
        memory_manager.stop()


async def batch_processing_example():
    """Example of batch processing multiple inputs."""
    print("\n🔄 Batch Processing Example")
    print("=" * 30)
    
    # Sample inputs
    test_inputs = [
        "Hello, how are you?",
        "What's the weather like?",
        "Tell me a joke",
        "What can you help me with?",
        "Goodbye"
    ]
    
    # Create a lightweight agent for batch processing
    agent_config = AgentConfig(
        name="batch_agent",
        model_name="microsoft/DialoGPT-small",
        max_tokens=50,
        temperature=0.5,
        memory_limit_gb=1.0
    )
    
    agent = SimpleConversationalAgent(agent_config)
    
    print("🔄 Initializing batch agent...")
    if not await agent.initialize():
        print("❌ Failed to initialize batch agent")
        return
    
    print("✅ Processing batch inputs...")
    
    results = []
    for i, input_text in enumerate(test_inputs, 1):
        print(f"📝 Processing {i}/{len(test_inputs)}: {input_text}")
        
        response = await agent.process(input_text)
        results.append({
            "input": input_text,
            "output": response.content,
            "processing_time_ms": response.processing_time_ms,
            "error": response.error
        })
        
        if response.error:
            print(f"   ❌ Error: {response.error}")
        else:
            print(f"   ✅ Response: {response.content[:50]}...")
    
    # Summary
    successful = len([r for r in results if not r["error"]])
    avg_time = sum(r["processing_time_ms"] for r in results if not r["error"]) / max(successful, 1)
    
    print(f"\n📊 Batch Processing Summary:")
    print(f"  • Total Inputs: {len(test_inputs)}")
    print(f"  • Successful: {successful}")
    print(f"  • Failed: {len(test_inputs) - successful}")
    print(f"  • Average Processing Time: {avg_time:.1f}ms")


if __name__ == "__main__":
    import contextlib
    
    try:
        # Run main example
        asyncio.run(main())
        
        # Run batch processing example
        print("\n" + "=" * 50)
        asyncio.run(batch_processing_example())
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        sys.exit(1)