"""
Query Agent - Handles query understanding and processing
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base_agent import BaseAgent


@dataclass
class QueryIntent:
    """Query intent analysis"""
    intent_type: str
    confidence: float
    entities: List[str]
    keywords: List[str]
    complexity: str


@dataclass
class QueryExpansion:
    """Query expansion results"""
    original_query: str
    expanded_queries: List[str]
    expansion_type: str


class QueryAgent(BaseAgent):
    """
    Agent responsible for query understanding and processing
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    async def analyze_query(self, query: str) -> QueryIntent:
        """
        Analyze query to understand intent and extract information
        
        Args:
            query: The user query
            
        Returns:
            QueryIntent with analysis results
        """
        prompt = f"""
        Analyze the following query and extract:
        1. Intent type (factual, analytical, creative, etc.)
        2. Named entities mentioned
        3. Key terms and concepts
        4. Query complexity level
        
        Query: {query}
        
        Provide analysis in structured format.
        """
        
        # For now, return a simplified analysis
        return QueryIntent(
            intent_type="factual",
            confidence=0.8,
            entities=[],
            keywords=query.lower().split(),
            complexity="medium"
        )
    
    async def detect_intent(self, query: str) -> str:
        """
        Detect the intent of a query
        
        Args:
            query: The user query
            
        Returns:
            Intent type as string
        """
        intents = [
            "factual", "analytical", "creative", "comparative",
            "procedural", "opinion", "clarification"
        ]
        
        # Simple keyword-based intent detection
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["how", "what", "when", "where"]):
            return "factual"
        elif any(word in query_lower for word in ["compare", "difference", "vs"]):
            return "comparative"
        elif any(word in query_lower for word in ["why", "explain", "analyze"]):
            return "analytical"
        else:
            return "factual"
    
    async def expand_query(self, query: str) -> QueryExpansion:
        """
        Expand query with related terms and synonyms
        
        Args:
            query: Original query
            
        Returns:
            QueryExpansion with expanded queries
        """
        prompt = f"""
        Expand the following query with related terms, synonyms, and alternative phrasings:
        
        Original query: {query}
        
        Provide 3-5 expanded versions that maintain the original intent.
        """
        
        # For now, return simple expansion
        expanded = [
            query,
            f"{query} information",
            f"details about {query}"
        ]
        
        return QueryExpansion(
            original_query=query,
            expanded_queries=expanded,
            expansion_type="synonym"
        )
    
    async def route_query(self, query: str) -> str:
        """
        Route query to appropriate processing strategy
        
        Args:
            query: The user query
            
        Returns:
            Strategy name
        """
        intent = await self.detect_intent(query)
        
        routing_map = {
            "factual": "semantic_search",
            "analytical": "hybrid_search",
            "creative": "generative_search",
            "comparative": "comparative_search"
        }
        
        return routing_map.get(intent, "semantic_search")
    
    async def analyze_semantic_content(self, query: str) -> Dict[str, Any]:
        """
        Analyze semantic content of the query
        
        Args:
            query: The user query
            
        Returns:
            Semantic analysis results
        """
        # Extract key concepts and relationships
        analysis = {
            "concepts": [],
            "relationships": [],
            "temporal_indicators": [],
            "spatial_indicators": []
        }
        
        # Simple analysis based on keywords
        words = query.lower().split()
        
        # Detect temporal indicators
        temporal_words = ["today", "yesterday", "recent", "historical"]
        analysis["temporal_indicators"] = [
            word for word in words if word in temporal_words
        ]
        
        # Detect spatial indicators
        spatial_words = ["location", "place", "area", "region"]
        analysis["spatial_indicators"] = [
            word for word in words if word in spatial_words
        ]
        
        return analysis
    
    async def extract_entities(self, query: str) -> List[str]:
        """
        Extract named entities from query
        
        Args:
            query: The user query
            
        Returns:
            List of extracted entities
        """
        # Simple entity extraction (in production, use NER models)
        entities = []
        
        # Look for capitalized words (potential entities)
        words = query.split()
        for word in words:
            if word[0].isupper() and len(word) > 1:
                entities.append(word)
        
        return entities
    
    async def process(self, input_data: Any) -> Any:
        """Main processing method"""
        if isinstance(input_data, str):
            return await self.analyze_query(input_data)
        elif isinstance(input_data, dict):
            query = input_data.get("query", "")
            analysis_type = input_data.get("analysis_type", "full")
            
            if analysis_type == "intent":
                return await self.detect_intent(query)
            elif analysis_type == "expansion":
                return await self.expand_query(query)
            elif analysis_type == "routing":
                return await self.route_query(query)
            elif analysis_type == "semantic":
                return await self.analyze_semantic_content(query)
            elif analysis_type == "entities":
                return await self.extract_entities(query)
            else:
                return await self.analyze_query(query)
        else:
            raise ValueError("Input must be string or dictionary")
    
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
    
    def _select_expansion_strategy(self, query: str) -> str:
        """Select appropriate expansion strategy"""
        if len(query.split()) <= 2:
            return "synonym"
        elif any(word in query.lower() for word in ["how", "what", "why"]):
            return "question_focused"
        else:
            return "concept_based" 