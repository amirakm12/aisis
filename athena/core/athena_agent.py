"""
Athena Agent - Main Orchestrator for Agentic Research System

The central intelligence that coordinates research activities across multiple
specialized agents and provides a unified interface for complex research tasks.
"""

import asyncio
import uuid
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import Tool

from .base_agent import BaseAgent, ResearchTask, ResearchResult
from .research_orchestrator import ResearchOrchestrator
from ..agents.web_researcher import WebResearchAgent
from ..agents.academic_researcher import AcademicResearchAgent  
from ..agents.data_analyst import DataAnalystAgent
from ..agents.synthesis_agent import SynthesisAgent


class AthenaAgent:
    """
    Main Athena Agent - The central intelligence of the research system.
    
    Capabilities:
    - Natural language research query processing
    - Multi-agent coordination and task distribution
    - Intelligent research strategy planning
    - Results synthesis and presentation
    - Continuous learning and adaptation
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.config = config or {}
        self.logger = logging.getLogger("athena.main")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        # Memory for conversation context
        self.memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Initialize research orchestrator
        self.orchestrator = ResearchOrchestrator(
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            config=self.config.get("orchestrator", {})
        )
        
        # Research session state
        self.current_session_id = None
        self.research_sessions: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("Athena Agent initialized successfully")
    
    async def research(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        research_depth: str = "standard",  # quick, standard, deep
        max_agents: int = 4
    ) -> Dict[str, Any]:
        """
        Conduct comprehensive research on a given query.
        
        Args:
            query: The research question or topic
            context: Additional context and constraints
            research_depth: Level of research depth (quick, standard, deep)
            max_agents: Maximum number of agents to use
            
        Returns:
            Comprehensive research results with synthesis
        """
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        
        self.logger.info(f"Starting research session {session_id} for query: {query}")
        
        # Initialize session
        session = {
            "id": session_id,
            "query": query,
            "context": context or {},
            "research_depth": research_depth,
            "started_at": datetime.now(),
            "status": "in_progress",
            "results": {},
            "synthesis": None
        }
        self.research_sessions[session_id] = session
        
        try:
            # Analyze query and create research strategy
            strategy = await self._create_research_strategy(query, context, research_depth)
            session["strategy"] = strategy
            
            # Execute research plan
            research_results = await self.orchestrator.execute_research_plan(
                strategy, max_agents=max_agents
            )
            session["results"] = research_results
            
            # Synthesize findings
            synthesis = await self._synthesize_findings(query, research_results, strategy)
            session["synthesis"] = synthesis
            
            # Update session status
            session["status"] = "completed"
            session["completed_at"] = datetime.now()
            
            self.logger.info(f"Completed research session {session_id}")
            
            return {
                "session_id": session_id,
                "query": query,
                "strategy": strategy,
                "results": research_results,
                "synthesis": synthesis,
                "metadata": {
                    "research_depth": research_depth,
                    "agents_used": len(research_results),
                    "total_sources": sum(len(r.get("sources", [])) for r in research_results.values()),
                    "duration": (session["completed_at"] - session["started_at"]).total_seconds()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in research session {session_id}: {e}")
            session["status"] = "failed"
            session["error"] = str(e)
            
            return {
                "session_id": session_id,
                "query": query,
                "error": str(e),
                "status": "failed"
            }
    
    async def _create_research_strategy(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        research_depth: str
    ) -> Dict[str, Any]:
        """
        Create an intelligent research strategy based on the query.
        """
        strategy_prompt = f"""
        As Athena, an advanced AI research strategist, analyze this research query and create a comprehensive research strategy.
        
        Query: {query}
        Context: {context or 'None provided'}
        Research Depth: {research_depth}
        
        Create a research strategy that includes:
        1. Query analysis and key concepts identification
        2. Research objectives and sub-questions
        3. Recommended research approaches and agent types
        4. Expected information sources and types
        5. Success criteria and evaluation metrics
        
        Format your response as a structured strategy plan.
        """
        
        messages = [
            SystemMessage(content="You are Athena, an expert AI research strategist. Create comprehensive, actionable research strategies."),
            HumanMessage(content=strategy_prompt)
        ]
        
        response = await self.llm.agenerate([messages])
        strategy_text = response.generations[0][0].text
        
        # Parse strategy into structured format
        strategy = {
            "query_analysis": self._extract_query_analysis(strategy_text),
            "research_objectives": self._extract_objectives(strategy_text),
            "recommended_agents": self._recommend_agents(query, context, research_depth),
            "research_depth": research_depth,
            "estimated_duration": self._estimate_duration(research_depth),
            "raw_strategy": strategy_text
        }
        
        return strategy
    
    def _extract_query_analysis(self, strategy_text: str) -> Dict[str, Any]:
        """Extract query analysis from strategy text"""
        # This would be enhanced with more sophisticated NLP parsing
        return {
            "key_concepts": [],  # Would extract key concepts
            "query_type": "general",  # Would classify query type
            "complexity": "medium",  # Would assess complexity
            "domain": "general"  # Would identify domain
        }
    
    def _extract_objectives(self, strategy_text: str) -> List[str]:
        """Extract research objectives from strategy text"""
        # This would be enhanced with more sophisticated parsing
        return [
            "Gather comprehensive information on the topic",
            "Identify key sources and references",
            "Analyze different perspectives and viewpoints",
            "Synthesize findings into coherent insights"
        ]
    
    def _recommend_agents(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
        research_depth: str
    ) -> List[str]:
        """Recommend which agents should be used for this research"""
        agents = ["web_researcher"]  # Always include web research
        
        # Add academic research for scholarly topics
        academic_keywords = ["research", "study", "analysis", "theory", "academic", "scientific"]
        if any(keyword in query.lower() for keyword in academic_keywords):
            agents.append("academic_researcher")
        
        # Add data analysis for quantitative queries
        data_keywords = ["data", "statistics", "numbers", "trends", "analysis", "metrics"]
        if any(keyword in query.lower() for keyword in data_keywords):
            agents.append("data_analyst")
        
        # Always include synthesis for comprehensive research
        if research_depth in ["standard", "deep"]:
            agents.append("synthesis_agent")
        
        return agents
    
    def _estimate_duration(self, research_depth: str) -> float:
        """Estimate research duration in minutes"""
        duration_map = {
            "quick": 2.0,
            "standard": 5.0,
            "deep": 10.0
        }
        return duration_map.get(research_depth, 5.0)
    
    async def _synthesize_findings(
        self,
        query: str,
        research_results: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Synthesize research findings into a comprehensive response.
        """
        synthesis_prompt = f"""
        As Athena, synthesize the following research findings into a comprehensive, well-structured response.
        
        Original Query: {query}
        Research Strategy: {strategy.get('raw_strategy', 'N/A')}
        
        Research Findings:
        {self._format_results_for_synthesis(research_results)}
        
        Provide a synthesis that includes:
        1. Executive Summary
        2. Key Findings
        3. Supporting Evidence
        4. Different Perspectives
        5. Conclusions and Implications
        6. Recommendations for Further Research
        
        Make the synthesis comprehensive, well-structured, and actionable.
        """
        
        messages = [
            SystemMessage(content="You are Athena, an expert AI research synthesizer. Create comprehensive, insightful syntheses of research findings."),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = await self.llm.agenerate([messages])
        synthesis_text = response.generations[0][0].text
        
        return {
            "executive_summary": self._extract_executive_summary(synthesis_text),
            "key_findings": self._extract_key_findings(synthesis_text),
            "evidence": self._extract_evidence(research_results),
            "conclusions": self._extract_conclusions(synthesis_text),
            "recommendations": self._extract_recommendations(synthesis_text),
            "full_synthesis": synthesis_text,
            "confidence_score": self._calculate_confidence(research_results)
        }
    
    def _format_results_for_synthesis(self, research_results: Dict[str, Any]) -> str:
        """Format research results for synthesis prompt"""
        formatted = []
        for agent_id, result in research_results.items():
            if isinstance(result, dict) and "content" in result:
                formatted.append(f"\n{agent_id.upper()} FINDINGS:\n{result['content']}")
        return "\n".join(formatted)
    
    def _extract_executive_summary(self, synthesis_text: str) -> str:
        """Extract executive summary from synthesis"""
        # This would be enhanced with more sophisticated parsing
        lines = synthesis_text.split('\n')
        summary_start = -1
        for i, line in enumerate(lines):
            if 'executive summary' in line.lower():
                summary_start = i + 1
                break
        
        if summary_start > -1:
            summary_lines = []
            for line in lines[summary_start:]:
                if line.strip() and not any(header in line.lower() for header in ['key findings', 'supporting evidence']):
                    summary_lines.append(line.strip())
                elif line.strip() and any(header in line.lower() for header in ['key findings', 'supporting evidence']):
                    break
            return ' '.join(summary_lines)
        
        return synthesis_text.split('\n')[0] if synthesis_text else "No summary available"
    
    def _extract_key_findings(self, synthesis_text: str) -> List[str]:
        """Extract key findings from synthesis"""
        # This would be enhanced with more sophisticated parsing
        return ["Key finding 1", "Key finding 2", "Key finding 3"]
    
    def _extract_evidence(self, research_results: Dict[str, Any]) -> List[str]:
        """Extract evidence sources from research results"""
        evidence = []
        for agent_id, result in research_results.items():
            if isinstance(result, dict) and "sources" in result:
                evidence.extend(result["sources"])
        return list(set(evidence))  # Remove duplicates
    
    def _extract_conclusions(self, synthesis_text: str) -> List[str]:
        """Extract conclusions from synthesis"""
        return ["Conclusion 1", "Conclusion 2"]
    
    def _extract_recommendations(self, synthesis_text: str) -> List[str]:
        """Extract recommendations from synthesis"""
        return ["Recommendation 1", "Recommendation 2"]
    
    def _calculate_confidence(self, research_results: Dict[str, Any]) -> float:
        """Calculate overall confidence score for the research"""
        if not research_results:
            return 0.0
        
        confidences = []
        for result in research_results.values():
            if isinstance(result, dict) and "confidence" in result:
                confidences.append(result["confidence"])
        
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def get_session_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of a research session"""
        sid = session_id or self.current_session_id
        if sid and sid in self.research_sessions:
            return self.research_sessions[sid]
        return {"error": "Session not found"}
    
    def list_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent research sessions"""
        sessions = sorted(
            self.research_sessions.values(),
            key=lambda x: x.get("started_at", datetime.min),
            reverse=True
        )[:limit]
        
        return [
            {
                "id": s["id"],
                "query": s["query"],
                "status": s["status"],
                "started_at": s["started_at"].isoformat(),
                "research_depth": s.get("research_depth", "standard")
            }
            for s in sessions
        ]
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up Athena Agent")
        if self.orchestrator:
            await self.orchestrator.cleanup()
        self.research_sessions.clear()