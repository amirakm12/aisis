#!/usr/bin/env python3
"""
Athena Agentic Research System Demo

This script demonstrates the capabilities of the Athena agentic research system
by running example research queries and showing the results.
"""

import asyncio
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

from athena import AthenaAgent

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def demo_research_query(athena: AthenaAgent, query: str, research_depth: str = "standard"):
    """Demonstrate a research query"""
    print(f"\n{'='*80}")
    print(f"RESEARCH QUERY: {query}")
    print(f"RESEARCH DEPTH: {research_depth}")
    print(f"{'='*80}")
    
    try:
        # Conduct research
        result = await athena.research(
            query=query,
            research_depth=research_depth,
            max_agents=3
        )
        
        # Display results
        if result.get("status") == "failed":
            print(f"❌ Research failed: {result.get('error')}")
            return
        
        print(f"✅ Research completed successfully!")
        print(f"Session ID: {result['session_id']}")
        
        # Show metadata
        metadata = result.get("metadata", {})
        print(f"\n📊 RESEARCH METADATA:")
        print(f"  • Research Depth: {metadata.get('research_depth', 'N/A')}")
        print(f"  • Agents Used: {metadata.get('agents_used', 0)}")
        print(f"  • Total Sources: {metadata.get('total_sources', 0)}")
        print(f"  • Duration: {metadata.get('duration', 0):.2f} seconds")
        
        # Show synthesis summary
        synthesis = result.get("synthesis", {})
        if synthesis:
            print(f"\n📝 EXECUTIVE SUMMARY:")
            executive_summary = synthesis.get("executive_summary", "No summary available")
            print(f"{executive_summary[:500]}...")
            
            print(f"\n🔍 KEY FINDINGS:")
            key_findings = synthesis.get("key_findings", [])
            for i, finding in enumerate(key_findings[:3], 1):
                print(f"  {i}. {finding}")
            
            print(f"\n📈 CONFIDENCE SCORE: {synthesis.get('confidence_score', 0):.2f}/1.0")
        
        # Show some research results
        results = result.get("results", {})
        if results:
            print(f"\n🔬 AGENT RESULTS:")
            for agent_id, agent_result in list(results.items())[:2]:  # Show first 2 agents
                content = agent_result.get("content", "")
                confidence = agent_result.get("confidence", 0)
                sources_count = len(agent_result.get("sources", []))
                
                print(f"\n  📋 {agent_id.upper()}:")
                print(f"    Confidence: {confidence:.2f}")
                print(f"    Sources: {sources_count}")
                print(f"    Content: {content[:200]}...")
        
    except Exception as e:
        print(f"❌ Error during research: {e}")
        logger.error(f"Research failed for query '{query}': {e}")


async def main():
    """Main demo function"""
    print("🔬 Athena Agentic Research System Demo")
    print("=====================================")
    
    # Get API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables.")
        print("   Some features may not work properly without API keys.")
        print("   Please set your API keys in a .env file or environment variables.\n")
    
    # Initialize Athena
    print("🚀 Initializing Athena Agent...")
    athena = AthenaAgent(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        config={
            "orchestrator": {
                "web_researcher": {"max_search_results": 5},
                "academic_researcher": {"max_papers": 5},
                "data_analyst": {"max_data_points": 1000}
            }
        }
    )
    
    # Demo queries with different research depths
    demo_queries = [
        {
            "query": "What are the latest developments in artificial intelligence and machine learning?",
            "depth": "quick"
        },
        {
            "query": "Climate change impact on global agriculture and food security",
            "depth": "standard"
        },
        {
            "query": "Quantum computing research advances and practical applications",
            "depth": "standard"
        }
    ]
    
    # Run demo queries
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n🧪 Demo {i}/{len(demo_queries)}")
        await demo_research_query(
            athena, 
            demo["query"], 
            demo["depth"]
        )
        
        # Add delay between queries to avoid rate limiting
        if i < len(demo_queries):
            print("\n⏳ Waiting 5 seconds before next query...")
            await asyncio.sleep(5)
    
    # Show system status
    print(f"\n{'='*80}")
    print("📊 SYSTEM STATUS")
    print(f"{'='*80}")
    
    # Get recent sessions
    sessions = athena.list_sessions(limit=5)
    print(f"\n📋 Recent Research Sessions ({len(sessions)}):")
    for session in sessions:
        status_emoji = "✅" if session["status"] == "completed" else "❌"
        print(f"  {status_emoji} {session['query'][:60]}...")
        print(f"     Status: {session['status']} | Depth: {session['research_depth']}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up resources...")
    await athena.cleanup()
    
    print(f"\n✨ Demo completed successfully!")
    print("Thank you for trying Athena Agentic Research System!")


def create_env_template():
    """Create a template .env file"""
    env_template = """# Athena Agentic Research System Configuration
# Copy this file to .env and fill in your API keys

# OpenAI API Key (required for most features)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional, for additional AI capabilities)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google Search API Key (optional, for enhanced web search)
GOOGLE_SEARCH_API_KEY=your_google_search_api_key_here

# Other optional configurations
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=4
DEFAULT_RESEARCH_DEPTH=standard
"""
    
    if not os.path.exists(".env"):
        with open(".env.template", "w") as f:
            f.write(env_template)
        print("📝 Created .env.template file. Please copy to .env and configure your API keys.")


if __name__ == "__main__":
    print("🔬 Athena Agentic Research System")
    print("==================================")
    
    # Create environment template if needed
    create_env_template()
    
    # Run the demo
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.error(f"Demo failed: {e}")