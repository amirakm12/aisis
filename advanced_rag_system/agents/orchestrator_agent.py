"""
Orchestrator Agent - Coordinates the entire RAG workflow
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent
from ..retrievers.hybrid_retriever import HybridRetriever
from ..utils.config import RAGConfig


@dataclass
class QueryPlan:
    """Plan for processing a query"""
    query_type: str
    retrieval_strategy: str
    reranking_enabled: bool
    max_sources: int
    context_requirements: Dict[str, Any]


class OrchestratorAgent(BaseAgent):
    """
    Orchestrator agent that coordinates the entire RAG workflow
    """
    
    def __init__(self, retriever: HybridRetriever, config: RAGConfig):
        super().__init__(config)
        self.retriever = retriever
    
    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline
        
        Args:
            query: The user query
            context: Additional context for the query
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        # Step 1: Analyze query
        query_analysis = await self._analyze_query(query, context)
        
        # Step 2: Create execution plan
        execution_plan = await self._create_execution_plan(query_analysis)
        
        # Step 3: Retrieve information
        retrieval_result = await self._retrieve_information(
            query, execution_plan
        )
        
        # Step 4: Synthesize response
        response = await self._synthesize_response(
            query, retrieval_result, context
        )
        
        # Step 5: Validate and enhance response
        final_response = await self._validate_and_enhance_response(
            query, response, retrieval_result
        )
        
        return final_response
    
    async def _analyze_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze the query to understand intent and requirements"""
        prompt = f"""
        Analyze the following query and provide insights about:
        1. Query type (factual, analytical, creative, etc.)
        2. Required information depth
        3. Specific entities or topics mentioned
        4. Context requirements
        
        Query: {query}
        Context: {context or 'None'}
        
        Provide your analysis in JSON format.
        """
        
        response = await self.llm.generate(prompt)
        # Parse JSON response (simplified)
        return {
            "query_type": "factual",  # Default
            "entities": [],
            "depth_required": "medium",
            "context_needed": True
        }
    
    async def _create_execution_plan(
        self,
        query_analysis: Dict[str, Any]
    ) -> QueryPlan:
        """Create an execution plan based on query analysis"""
        return QueryPlan(
            query_type=query_analysis.get("query_type", "factual"),
            retrieval_strategy="hybrid",
            reranking_enabled=self.config.retriever.use_reranking,
            max_sources=self.config.retriever.top_k,
            context_requirements=query_analysis
        )
    
    async def _retrieve_information(
        self,
        query: str,
        execution_plan: QueryPlan
    ) -> Any:
        """Retrieve relevant information using the retriever"""
        return await self.retriever.retrieve(
            query,
            top_k=execution_plan.max_sources,
            use_reranking=execution_plan.reranking_enabled
        )
    
    async def _synthesize_response(
        self,
        query: str,
        retrieval_result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Synthesize a response based on retrieved information"""
        # Prepare context from retrieved documents
        context_text = ""
        if retrieval_result.documents:
            context_text = "\n\n".join([
                f"Source {i+1}: {doc.get('content', '')}"
                for i, doc in enumerate(retrieval_result.documents)
            ])
        
        prompt = f"""
        Based on the following information, provide a comprehensive answer to the query.
        
        Query: {query}
        Context: {context or 'None'}
        
        Retrieved Information:
        {context_text}
        
        Please provide a well-structured answer that:
        1. Directly addresses the query
        2. Uses information from the retrieved sources
        3. Acknowledges the sources used
        4. Maintains accuracy and relevance
        """
        
        answer = await self.llm.generate(prompt)
        
        return {
            "answer": answer,
            "sources": retrieval_result.documents,
            "confidence": retrieval_result.confidence,
            "total_tokens": len(answer.split()),
            "metadata": {
                "retrieval_strategy": "hybrid",
                "sources_used": len(retrieval_result.documents)
            }
        }
    
    async def _validate_and_enhance_response(
        self,
        query: str,
        response: Dict[str, Any],
        retrieval_result: Any
    ) -> Dict[str, Any]:
        """Validate and enhance the response"""
        # Check if response is adequate
        if not response["sources"]:
            response["answer"] = (
                "I don't have enough information to provide a complete answer. "
                "Please try rephrasing your query or providing more context."
            )
            response["confidence"] = 0.3
        
        # Enhance with additional context if needed
        if response["confidence"] < 0.7:
            response["answer"] += (
                "\n\nNote: This response has lower confidence. "
                "Consider providing more specific details in your query."
            )
        
        return response
    
    async def process(self, input_data: Any) -> Any:
        """Main processing method"""
        if isinstance(input_data, dict):
            query = input_data.get("query", "")
            context = input_data.get("context")
            return await self.process_query(query, context)
        else:
            return await self.process_query(str(input_data))
    
    def validate_input(self, input_data: Any) -> Any:
        """Validate input data"""
        if isinstance(input_data, str):
            if not input_data.strip():
                raise ValueError("Query cannot be empty")
        elif isinstance(input_data, dict):
            if "query" not in input_data:
                raise ValueError("Input must contain 'query' field")
        else:
            raise ValueError("Input must be string or dictionary")
        return input_data 