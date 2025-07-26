"""
Orchestrator Agent - Coordinates all other agents and manages the RAG workflow
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from loguru import logger

from ..utils.config import AgentConfig, LLMConfig
from .base_agent import BaseAgent
from .query_agent import QueryAgent
from .document_agent import DocumentAgent
from ..retrievers.hybrid_retriever import HybridRetriever

@dataclass
class QueryPlan:
    """Plan for processing a query"""
    query_type: str
    requires_expansion: bool
    requires_routing: bool
    retrieval_strategy: str
    confidence_threshold: float
    max_sources: int

class OrchestratorAgent(BaseAgent):
    """
    Orchestrator Agent that coordinates the entire RAG workflow
    
    Responsibilities:
    - Analyze incoming queries and create execution plans
    - Coordinate query and document agents
    - Manage retrieval strategies
    - Ensure quality and consistency of responses
    - Handle error recovery and fallback strategies
    """
    
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_config: LLMConfig,
        query_agent: QueryAgent,
        document_agent: DocumentAgent,
        hybrid_retriever: HybridRetriever
    ):
        super().__init__(agent_config, llm_config)
        self.query_agent = query_agent
        self.document_agent = document_agent
        self.hybrid_retriever = hybrid_retriever
        
        # Query processing templates
        self.query_analysis_template = """
        Analyze the following query and provide a structured analysis:
        
        Query: {query}
        Context: {context}
        
        Please provide:
        1. Query type (factual, analytical, comparison, summarization, etc.)
        2. Complexity level (simple, medium, complex)
        3. Required information types
        4. Suggested retrieval strategy
        5. Confidence in understanding (0-1)
        
        Format your response as JSON:
        {{
            "query_type": "...",
            "complexity": "...",
            "required_info": [...],
            "retrieval_strategy": "...",
            "confidence": 0.0
        }}
        """
        
        self.response_synthesis_template = """
        Based on the retrieved information, provide a comprehensive answer:
        
        Original Query: {query}
        Context: {context}
        Retrieved Information: {retrieved_info}
        Query Analysis: {query_analysis}
        
        Please provide:
        1. A clear, accurate answer
        2. Confidence level in the answer (0-1)
        3. Key supporting evidence
        4. Any limitations or uncertainties
        
        Format your response as JSON:
        {{
            "answer": "...",
            "confidence": 0.0,
            "evidence": [...],
            "limitations": "..."
        }}
        """
    
    async def initialize(self):
        """Initialize the orchestrator agent"""
        await super().initialize()
        logger.info("Orchestrator Agent initialized")
    
    async def process_query(
        self,
        query: str,
        context: Dict[str, Any],
        filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main method to process a query through the complete RAG pipeline
        
        Args:
            query: The user query
            context: Additional context
            filters: Retrieval filters
            
        Returns:
            Complete response with answer and metadata
        """
        try:
            # Step 1: Analyze the query
            query_analysis = await self._analyze_query(query, context)
            logger.info(f"Query analysis completed: {query_analysis['query_type']}")
            
            # Step 2: Create execution plan
            execution_plan = await self._create_execution_plan(query_analysis, context)
            logger.info(f"Execution plan created: {execution_plan.retrieval_strategy}")
            
            # Step 3: Process query with query agent if needed
            processed_query = query
            if execution_plan.requires_expansion:
                processed_query = await self.query_agent.expand_query(
                    query, context, query_analysis
                )
                logger.info("Query expansion completed")
            
            # Step 4: Route query if needed
            routing_info = {}
            if execution_plan.requires_routing:
                routing_info = await self.query_agent.route_query(
                    processed_query, context, query_analysis
                )
                logger.info(f"Query routing completed: {routing_info}")
            
            # Step 5: Retrieve relevant information
            retrieval_results = await self._retrieve_information(
                processed_query,
                execution_plan,
                filters,
                routing_info
            )
            logger.info(f"Retrieved {len(retrieval_results['documents'])} documents")
            
            # Step 6: Synthesize response
            response = await self._synthesize_response(
                query,
                context,
                query_analysis,
                retrieval_results
            )
            logger.info("Response synthesis completed")
            
            # Step 7: Validate and enhance response
            final_response = await self._validate_and_enhance_response(
                response,
                query_analysis,
                retrieval_results
            )
            
            return {
                "answer": final_response["answer"],
                "confidence": final_response["confidence"],
                "sources": retrieval_results["documents"],
                "query_analysis": query_analysis,
                "retrieval_metadata": retrieval_results["metadata"],
                "execution_plan": execution_plan.__dict__,
                "processing_steps": final_response.get("processing_steps", [])
            }
            
        except Exception as e:
            logger.error(f"Error in orchestrator processing: {str(e)}")
            return await self._handle_error(query, context, str(e))
    
    async def _analyze_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze the query to understand its characteristics"""
        try:
            prompt = self.query_analysis_template.format(
                query=query,
                context=context
            )
            
            response = await self.llm.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            # Parse JSON response
            import json
            analysis = json.loads(response.strip())
            
            # Add additional analysis
            analysis.update({
                "original_query": query,
                "context_provided": bool(context),
                "query_length": len(query),
                "estimated_complexity": self._estimate_complexity(query)
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            # Return default analysis
            return {
                "query_type": "general",
                "complexity": "medium",
                "required_info": ["general"],
                "retrieval_strategy": "hybrid",
                "confidence": 0.5,
                "original_query": query,
                "context_provided": bool(context),
                "query_length": len(query),
                "estimated_complexity": "medium"
            }
    
    async def _create_execution_plan(
        self,
        query_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> QueryPlan:
        """Create an execution plan based on query analysis"""
        
        # Determine if query expansion is needed
        requires_expansion = (
            query_analysis.get("complexity") in ["medium", "complex"] or
            query_analysis.get("query_type") in ["analytical", "comparison"] or
            len(query_analysis.get("original_query", "")) < 20
        )
        
        # Determine if query routing is needed
        requires_routing = (
            query_analysis.get("query_type") in ["technical", "domain_specific"] or
            bool(context.get("domain")) or
            query_analysis.get("confidence", 0) < 0.7
        )
        
        # Select retrieval strategy
        retrieval_strategy = "hybrid"  # Default
        if query_analysis.get("query_type") == "factual":
            retrieval_strategy = "semantic"
        elif query_analysis.get("query_type") in ["comparison", "analytical"]:
            retrieval_strategy = "hybrid"
        elif query_analysis.get("complexity") == "simple":
            retrieval_strategy = "keyword"
        
        # Set confidence threshold
        confidence_threshold = 0.7
        if query_analysis.get("complexity") == "complex":
            confidence_threshold = 0.6
        elif query_analysis.get("complexity") == "simple":
            confidence_threshold = 0.8
        
        # Set max sources
        max_sources = 10
        if query_analysis.get("query_type") in ["comparison", "analytical"]:
            max_sources = 15
        elif query_analysis.get("complexity") == "simple":
            max_sources = 5
        
        return QueryPlan(
            query_type=query_analysis.get("query_type", "general"),
            requires_expansion=requires_expansion,
            requires_routing=requires_routing,
            retrieval_strategy=retrieval_strategy,
            confidence_threshold=confidence_threshold,
            max_sources=max_sources
        )
    
    async def _retrieve_information(
        self,
        query: str,
        execution_plan: QueryPlan,
        filters: Dict[str, Any],
        routing_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Retrieve relevant information based on the execution plan"""
        
        # Prepare retrieval parameters
        retrieval_params = {
            "query": query,
            "top_k": execution_plan.max_sources,
            "strategy": execution_plan.retrieval_strategy,
            "filters": filters,
            "confidence_threshold": execution_plan.confidence_threshold
        }
        
        # Add routing information to filters if available
        if routing_info:
            retrieval_params["filters"].update(routing_info.get("filters", {}))
        
        # Perform retrieval
        results = await self.hybrid_retriever.retrieve(**retrieval_params)
        
        return {
            "documents": results,
            "metadata": {
                "strategy_used": execution_plan.retrieval_strategy,
                "total_retrieved": len(results),
                "filters_applied": retrieval_params["filters"],
                "confidence_threshold": execution_plan.confidence_threshold
            }
        }
    
    async def _synthesize_response(
        self,
        original_query: str,
        context: Dict[str, Any],
        query_analysis: Dict[str, Any],
        retrieval_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize the final response from retrieved information"""
        
        # Prepare retrieved information for synthesis
        retrieved_info = []
        for doc in retrieval_results["documents"]:
            retrieved_info.append({
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
                "score": doc.get("score", 0.0)
            })
        
        prompt = self.response_synthesis_template.format(
            query=original_query,
            context=context,
            retrieved_info=retrieved_info[:5],  # Limit to top 5 for synthesis
            query_analysis=query_analysis
        )
        
        try:
            response = await self.llm.generate_response(
                prompt,
                temperature=0.5,
                max_tokens=1000
            )
            
            # Parse JSON response
            import json
            synthesis_result = json.loads(response.strip())
            
            return synthesis_result
            
        except Exception as e:
            logger.error(f"Error synthesizing response: {str(e)}")
            # Fallback response
            return {
                "answer": "I apologize, but I encountered an error while processing your query. Please try rephrasing your question.",
                "confidence": 0.1,
                "evidence": [],
                "limitations": "Error in response synthesis"
            }
    
    async def _validate_and_enhance_response(
        self,
        response: Dict[str, Any],
        query_analysis: Dict[str, Any],
        retrieval_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and enhance the response"""
        
        enhanced_response = response.copy()
        processing_steps = []
        
        # Check confidence level
        if response.get("confidence", 0) < 0.5:
            processing_steps.append("Low confidence detected - added uncertainty markers")
            enhanced_response["answer"] = f"Based on available information: {response['answer']}"
        
        # Add source citations if not present
        if not response.get("evidence"):
            top_sources = retrieval_results["documents"][:3]
            enhanced_response["evidence"] = [
                f"Source {i+1}: {doc.get('metadata', {}).get('title', 'Document')}"
                for i, doc in enumerate(top_sources)
            ]
            processing_steps.append("Added source citations")
        
        # Check for completeness based on query type
        if query_analysis.get("query_type") == "comparison" and "compare" not in response["answer"].lower():
            enhanced_response["answer"] += "\n\nNote: This appears to be a comparison query. Please let me know if you need a more detailed comparison."
            processing_steps.append("Added comparison guidance")
        
        # Add processing metadata
        enhanced_response["processing_steps"] = processing_steps
        enhanced_response["validation_passed"] = True
        
        return enhanced_response
    
    async def _handle_error(
        self,
        query: str,
        context: Dict[str, Any],
        error: str
    ) -> Dict[str, Any]:
        """Handle errors gracefully"""
        logger.error(f"Orchestrator error for query '{query}': {error}")
        
        return {
            "answer": "I apologize, but I encountered an error while processing your query. Please try again or rephrase your question.",
            "confidence": 0.0,
            "sources": [],
            "query_analysis": {"error": error},
            "retrieval_metadata": {"error": True},
            "execution_plan": {},
            "processing_steps": ["Error handling activated"]
        }
    
    def _estimate_complexity(self, query: str) -> str:
        """Estimate query complexity based on simple heuristics"""
        word_count = len(query.split())
        
        if word_count < 5:
            return "simple"
        elif word_count > 15:
            return "complex"
        else:
            return "medium"