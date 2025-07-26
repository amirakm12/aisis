#!/usr/bin/env python3
"""
Athena System Test Script

Simple tests to verify that the Athena system components work correctly.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all components can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from athena import AthenaAgent
        from athena.core.athena_agent import AthenaAgent as CoreAthena
        from athena.core.research_orchestrator import ResearchOrchestrator
        from athena.core.base_agent import BaseAgent, ResearchTask, ResearchResult
        from athena.agents.web_researcher import WebResearchAgent
        from athena.agents.academic_researcher import AcademicResearchAgent
        from athena.agents.data_analyst import DataAnalystAgent
        from athena.agents.synthesis_agent import SynthesisAgent
        
        print("✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_agent_initialization():
    """Test agent initialization without API keys"""
    print("\n🔍 Testing agent initialization...")
    
    try:
        # Test base components
        from athena.core.base_agent import ResearchTask
        from athena.agents.web_researcher import WebResearchAgent
        from athena.agents.academic_researcher import AcademicResearchAgent
        from athena.agents.data_analyst import DataAnalystAgent
        from athena.agents.synthesis_agent import SynthesisAgent
        
        # Create test task
        test_task = ResearchTask(
            id="test-task-1",
            query="Test query for system verification",
            context={"test": True}
        )
        
        print(f"✅ Created test task: {test_task.id}")
        
        # Initialize agents without API keys (should work for basic functionality)
        web_agent = WebResearchAgent(config={"max_search_results": 3})
        print(f"✅ Web Research Agent initialized: {web_agent.name}")
        
        academic_agent = AcademicResearchAgent(config={"max_papers": 3})
        print(f"✅ Academic Research Agent initialized: {academic_agent.name}")
        
        data_agent = DataAnalystAgent(config={"max_data_points": 100})
        print(f"✅ Data Analyst Agent initialized: {data_agent.name}")
        
        synthesis_agent = SynthesisAgent(config={"synthesis_depth": "basic"})
        print(f"✅ Synthesis Agent initialized: {synthesis_agent.name}")
        
        # Test capability checking
        print("\n🔍 Testing agent capabilities...")
        
        # Test web agent capabilities
        web_can_handle = web_agent.can_handle_task(test_task)
        print(f"  Web agent can handle test task: {web_can_handle}")
        
        # Test academic agent capabilities
        academic_task = ResearchTask(
            id="academic-test",
            query="Research paper analysis on machine learning",
            context={}
        )
        academic_can_handle = academic_agent.can_handle_task(academic_task)
        print(f"  Academic agent can handle academic task: {academic_can_handle}")
        
        # Test data agent capabilities
        data_task = ResearchTask(
            id="data-test",
            query="Statistical analysis of data trends",
            context={}
        )
        data_can_handle = data_agent.can_handle_task(data_task)
        print(f"  Data agent can handle data task: {data_can_handle}")
        
        # Test synthesis agent capabilities
        synthesis_task = ResearchTask(
            id="synthesis-test",
            query="Synthesize research findings from multiple sources",
            context={"sources": ["source1", "source2"]}
        )
        synthesis_can_handle = synthesis_agent.can_handle_task(synthesis_task)
        print(f"  Synthesis agent can handle synthesis task: {synthesis_can_handle}")
        
        print("✅ Agent initialization and capability testing successful!")
        return True
        
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False


async def test_basic_orchestration():
    """Test basic orchestration without API calls"""
    print("\n🔍 Testing basic orchestration...")
    
    try:
        from athena.core.research_orchestrator import ResearchOrchestrator
        
        # Initialize orchestrator without API keys
        orchestrator = ResearchOrchestrator(config={
            "web_researcher": {"max_search_results": 2},
            "academic_researcher": {"max_papers": 2}
        })
        
        print(f"✅ Research Orchestrator initialized with {len(orchestrator.agents)} agents")
        
        # Test agent status
        status = orchestrator.get_agent_status()
        print(f"✅ Retrieved agent status for {len(status)} agents")
        
        # Test orchestrator stats
        stats = orchestrator.get_orchestrator_stats()
        print(f"✅ Retrieved orchestrator stats: {stats['total_agents']} total agents")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic orchestration test failed: {e}")
        return False


def test_data_structures():
    """Test data structure creation and manipulation"""
    print("\n🔍 Testing data structures...")
    
    try:
        from athena.core.base_agent import ResearchTask, ResearchResult
        
        # Create test task
        task = ResearchTask(
            id="test-123",
            query="Test research query",
            context={"source": "test"},
            priority=1
        )
        
        print(f"✅ Created ResearchTask: {task.id}")
        print(f"   Query: {task.query}")
        print(f"   Created at: {task.created_at}")
        
        # Create test result
        result = ResearchResult(
            task_id=task.id,
            agent_id="test-agent",
            content="Test research findings",
            sources=["source1", "source2"],
            confidence=0.85,
            metadata={"test": True}
        )
        
        print(f"✅ Created ResearchResult: {result.task_id}")
        print(f"   Agent: {result.agent_id}")
        print(f"   Confidence: {result.confidence}")
        print(f"   Sources: {len(result.sources)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False


def test_configuration():
    """Test configuration handling"""
    print("\n🔍 Testing configuration...")
    
    try:
        from athena.agents.web_researcher import WebResearchAgent
        
        # Test with custom configuration
        config = {
            "max_search_results": 5,
            "max_content_length": 1000,
            "search_timeout": 15
        }
        
        agent = WebResearchAgent(config=config)
        
        print(f"✅ Agent configured with custom settings:")
        print(f"   Max search results: {agent.max_search_results}")
        print(f"   Max content length: {agent.max_content_length}")
        print(f"   Search timeout: {agent.search_timeout}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def run_all_tests():
    """Run all tests"""
    print("🧪 Athena System Test Suite")
    print("===========================")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Import Test", test_imports),
        ("Agent Initialization", test_agent_initialization),
        ("Data Structures", test_data_structures),
        ("Configuration", test_configuration),
        ("Basic Orchestration", test_basic_orchestration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Athena system is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    print("🔬 Athena System Verification")
    print("=============================")
    
    try:
        success = asyncio.run(run_all_tests())
        exit_code = 0 if success else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        sys.exit(1)