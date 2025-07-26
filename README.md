# Athena - Agentic Research System

<div align="center">
  <h3>🔬 Advanced AI-Powered Research Platform</h3>
  <p><em>Autonomous multi-agent research system for comprehensive information gathering and analysis</em></p>
</div>

## 🌟 Overview

Athena is a sophisticated agentic research system that leverages multiple specialized AI agents to conduct comprehensive research on any topic. The system orchestrates different types of research agents that work together to gather, analyze, and synthesize information from various sources.

### ✨ Key Features

- **🤖 Multi-Agent Architecture**: Specialized agents for different research tasks
- **🌐 Web Research**: Comprehensive web search and content analysis
- **📚 Academic Research**: Scholarly paper analysis from ArXiv and Google Scholar
- **📊 Data Analysis**: Statistical analysis and visualization capabilities
- **🔄 Intelligent Synthesis**: Combines findings from multiple sources
- **⚡ Parallel Processing**: Concurrent agent execution for faster results
- **🎯 Adaptive Strategies**: Dynamic research planning based on query analysis
- **📈 Confidence Scoring**: Quality assessment of research findings

## 🏗️ Architecture

### Core Components

1. **Athena Agent** - Main orchestrator and interface
2. **Research Orchestrator** - Manages agent coordination and task distribution
3. **Specialized Research Agents**:
   - **Web Research Agent** - Internet search and content extraction
   - **Academic Research Agent** - Scholarly paper analysis
   - **Data Analyst Agent** - Statistical analysis and visualization
   - **Synthesis Agent** - Multi-source information synthesis

### Agent Capabilities

| Agent | Primary Functions | Data Sources |
|-------|------------------|--------------|
| Web Research | Search engines, web scraping, news analysis | DuckDuckGo, web pages, news sites |
| Academic Research | Paper analysis, citation tracking, literature review | ArXiv, Google Scholar |
| Data Analyst | Statistical analysis, visualization, trend analysis | Generated/provided datasets |
| Synthesis | Information integration, contradiction resolution | All agent outputs |

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- OpenAI API key (required)
- Anthropic API key (optional)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd athena-research-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.template .env
   # Edit .env file with your API keys
   ```

4. **Run the demo**:
   ```bash
   python demo.py
   ```

## 📖 Usage

### Basic Usage

```python
import asyncio
from athena import AthenaAgent

async def main():
    # Initialize Athena
    athena = AthenaAgent(
        openai_api_key="your-openai-api-key",
        anthropic_api_key="your-anthropic-api-key"  # optional
    )
    
    # Conduct research
    result = await athena.research(
        query="What are the latest developments in quantum computing?",
        research_depth="standard",  # quick, standard, deep
        max_agents=4
    )
    
    # Access results
    print(f"Research completed: {result['synthesis']['executive_summary']}")
    
    # Cleanup
    await athena.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Configuration

```python
config = {
    "orchestrator": {
        "web_researcher": {
            "max_search_results": 10,
            "max_content_length": 5000,
            "search_timeout": 30
        },
        "academic_researcher": {
            "max_papers": 15,
            "min_citation_count": 5
        },
        "data_analyst": {
            "max_data_points": 10000,
            "visualization_format": "png"
        },
        "synthesis_agent": {
            "synthesis_depth": "comprehensive",
            "include_contradictions": True
        }
    }
}

athena = AthenaAgent(
    openai_api_key="your-key",
    config=config
)
```

## 🔍 Research Depths

- **Quick** (~2 minutes): Basic web search and summary
- **Standard** (~5 minutes): Multi-agent research with synthesis
- **Deep** (~10 minutes): Comprehensive analysis with academic sources

## 📊 Output Structure

```python
{
    "session_id": "unique-session-id",
    "query": "research question",
    "strategy": {
        "query_analysis": {...},
        "research_objectives": [...],
        "recommended_agents": [...]
    },
    "results": {
        "web_researcher_task_id": {
            "content": "research findings",
            "sources": ["url1", "url2"],
            "confidence": 0.85
        },
        # ... other agent results
    },
    "synthesis": {
        "executive_summary": "brief overview",
        "key_findings": ["finding1", "finding2"],
        "conclusions": ["conclusion1", "conclusion2"],
        "confidence_score": 0.82
    },
    "metadata": {
        "research_depth": "standard",
        "agents_used": 3,
        "total_sources": 15,
        "duration": 245.6
    }
}
```

## 🛠️ API Reference

### AthenaAgent

#### `research(query, context=None, research_depth="standard", max_agents=4)`

Conduct comprehensive research on a query.

**Parameters:**
- `query` (str): The research question or topic
- `context` (dict, optional): Additional context and constraints
- `research_depth` (str): Level of research depth ("quick", "standard", "deep")
- `max_agents` (int): Maximum number of agents to use

**Returns:**
- `dict`: Comprehensive research results with synthesis

#### `get_session_status(session_id=None)`

Get the status of a research session.

#### `list_sessions(limit=10)`

List recent research sessions.

### Research Orchestrator

#### `execute_research_plan(strategy, max_agents=4)`

Execute a research plan using multiple agents.

#### `get_agent_status()`

Get the status of all agents.

## 🔧 Configuration Options

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_SEARCH_API_KEY=your_google_search_api_key
LOG_LEVEL=INFO
MAX_CONCURRENT_AGENTS=4
DEFAULT_RESEARCH_DEPTH=standard
```

### Agent Configuration

Each agent can be configured through the main config dictionary:

```python
{
    "web_researcher": {
        "max_search_results": 10,
        "max_content_length": 5000,
        "search_timeout": 30
    },
    "academic_researcher": {
        "max_papers": 10,
        "max_abstract_length": 2000,
        "min_citation_count": 5
    },
    "data_analyst": {
        "max_data_points": 10000,
        "visualization_format": "png",
        "statistical_significance": 0.05
    },
    "synthesis_agent": {
        "max_input_length": 15000,
        "synthesis_depth": "comprehensive",
        "include_contradictions": True
    }
}
```

## 📈 Performance & Scalability

- **Concurrent Processing**: Agents run in parallel for faster results
- **Intelligent Caching**: Avoids redundant API calls
- **Rate Limiting**: Respects API rate limits
- **Resource Management**: Automatic cleanup and memory management
- **Error Handling**: Robust error handling and recovery

## 🧪 Examples

### Research Examples

1. **Technology Research**:
   ```python
   result = await athena.research(
       "Latest developments in artificial intelligence and machine learning",
       research_depth="deep"
   )
   ```

2. **Academic Research**:
   ```python
   result = await athena.research(
       "Climate change impact on global agriculture research papers",
       research_depth="standard"
   )
   ```

3. **Data Analysis**:
   ```python
   result = await athena.research(
       "Statistical analysis of cryptocurrency market trends",
       research_depth="standard"
   )
   ```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Development Setup

1. Clone the repository
2. Install development dependencies: `pip install -r requirements-dev.txt`
3. Run tests: `pytest`
4. Follow the code style guidelines

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Full documentation available in the `/docs` folder
- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Join our community discussions

## 🔮 Roadmap

- [ ] Additional research agents (social media, financial data, etc.)
- [ ] Web-based interface
- [ ] Integration with more academic databases
- [ ] Advanced visualization capabilities
- [ ] Real-time collaborative research
- [ ] Plugin system for custom agents

## 🙏 Acknowledgments

- OpenAI for GPT models
- Anthropic for Claude models
- LangChain for agent framework
- All open-source contributors

---

<div align="center">
  <p><strong>Built with ❤️ for researchers, analysts, and curious minds</strong></p>
  <p><em>Athena - Wisdom through intelligent research</em></p>
</div>