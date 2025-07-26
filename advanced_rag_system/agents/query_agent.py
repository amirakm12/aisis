"""
Query Agent - Handles query understanding, expansion, and routing
"""

import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

from ..utils.config import AgentConfig, LLMConfig
from .base_agent import BaseAgent
from ..embeddings.embedding_manager import EmbeddingManager

class QueryAgent(BaseAgent):
    """
    Query Agent responsible for:
    - Query understanding and analysis
    - Query expansion and reformulation
    - Query routing to appropriate knowledge domains
    - Intent detection and classification
    """
    
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_config: LLMConfig,
        embedding_manager: EmbeddingManager
    ):
        super().__init__(agent_config, llm_config)
        self.embedding_manager = embedding_manager
        
        # Query expansion templates
        self.expansion_template = """
        Given the following query, generate 3-5 alternative formulations that capture the same intent:
        
        Original Query: {query}
        Context: {context}
        Query Analysis: {analysis}
        
        Generate variations that:
        1. Use synonyms and alternative terminology
        2. Rephrase the question structure
        3. Add relevant domain-specific terms
        4. Include related concepts
        
        Format as a JSON list:
        ["expanded query 1", "expanded query 2", ...]
        """
        
        # Query routing template
        self.routing_template = """
        Analyze this query and determine the most relevant knowledge domains:
        
        Query: {query}
        Context: {context}
        Analysis: {analysis}
        
        Available domains: {domains}
        
        Provide:
        1. Primary domain (most relevant)
        2. Secondary domains (if applicable)
        3. Confidence score (0-1)
        4. Suggested filters for retrieval
        
        Format as JSON:
        {{
            "primary_domain": "...",
            "secondary_domains": [...],
            "confidence": 0.0,
            "filters": {{...}}
        }}
        """
        
        # Intent detection template
        self.intent_template = """
        Classify the intent of this query:
        
        Query: {query}
        Context: {context}
        
        Possible intents:
        - information_seeking: Looking for factual information
        - comparison: Comparing multiple items/concepts
        - explanation: Seeking detailed explanation
        - how_to: Step-by-step instructions
        - troubleshooting: Problem-solving
        - definition: Word/concept definitions
        - analysis: Deep analysis or insights
        - synthesis: Combining information from multiple sources
        
        Format as JSON:
        {{
            "primary_intent": "...",
            "confidence": 0.0,
            "secondary_intents": [...],
            "complexity_level": "simple|medium|complex"
        }}
        """
    
    async def process(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main processing method"""
        return await self.analyze_query(query, context)
    
    async def analyze_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Comprehensive query analysis"""
        try:
            # Detect intent
            intent_result = await self.detect_intent(query, context)
            
            # Analyze semantic content
            semantic_analysis = await self.analyze_semantic_content(query)
            
            # Extract entities and keywords
            entities = await self.extract_entities(query)
            
            return {
                "intent": intent_result,
                "semantic_analysis": semantic_analysis,
                "entities": entities,
                "original_query": query,
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error in query analysis: {str(e)}")
            raise
    
    async def detect_intent(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect the intent of the query"""
        try:
            prompt = self.intent_template.format(
                query=query,
                context=context
            )
            
            response = await self.llm.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=300
            )
            
            import json
            intent_result = json.loads(response.strip())
            
            # Add heuristic-based validation
            intent_result = self._validate_intent(query, intent_result)
            
            return intent_result
            
        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            # Fallback intent detection
            return {
                "primary_intent": "information_seeking",
                "confidence": 0.5,
                "secondary_intents": [],
                "complexity_level": "medium"
            }
    
    async def expand_query(
        self,
        query: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """Expand the query for better retrieval"""
        if not self.agent_config.enable_query_expansion:
            return query
        
        try:
            prompt = self.expansion_template.format(
                query=query,
                context=context,
                analysis=analysis
            )
            
            response = await self.llm.generate_response(
                prompt,
                temperature=0.7,
                max_tokens=200
            )
            
            import json
            expanded_queries = json.loads(response.strip())
            
            # Combine original with best expansion
            if expanded_queries and len(expanded_queries) > 0:
                # Select the best expansion based on semantic similarity
                best_expansion = await self._select_best_expansion(
                    query, expanded_queries
                )
                return f"{query} {best_expansion}"
            
            return query
            
        except Exception as e:
            logger.error(f"Error expanding query: {str(e)}")
            return query
    
    async def route_query(
        self,
        query: str,
        context: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route query to appropriate knowledge domains"""
        if not self.agent_config.enable_query_routing:
            return {}
        
        try:
            # Get available domains (this would come from system config)
            available_domains = [
                "general", "technical", "business", "scientific",
                "legal", "medical", "educational", "creative"
            ]
            
            prompt = self.routing_template.format(
                query=query,
                context=context,
                analysis=analysis,
                domains=available_domains
            )
            
            response = await self.llm.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=300
            )
            
            import json
            routing_result = json.loads(response.strip())
            
            # Validate and enhance routing result
            routing_result = self._validate_routing(routing_result, available_domains)
            
            return routing_result
            
        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            return {
                "primary_domain": "general",
                "secondary_domains": [],
                "confidence": 0.5,
                "filters": {}
            }
    
    async def analyze_semantic_content(self, query: str) -> Dict[str, Any]:
        """Analyze semantic content of the query"""
        try:
            # Get query embedding
            query_embedding = await self.embedding_manager.embed_text(query)
            
            # Analyze query structure
            words = query.split()
            
            # Simple semantic features
            question_words = ["what", "how", "why", "when", "where", "who", "which"]
            has_question_word = any(word.lower() in question_words for word in words)
            
            comparison_words = ["vs", "versus", "compare", "difference", "better", "best"]
            is_comparison = any(word.lower() in comparison_words for word in words)
            
            temporal_words = ["recent", "latest", "new", "old", "current", "past"]
            has_temporal = any(word.lower() in temporal_words for word in words)
            
            return {
                "embedding_vector": query_embedding.tolist() if query_embedding is not None else [],
                "word_count": len(words),
                "has_question_word": has_question_word,
                "is_comparison": is_comparison,
                "has_temporal_aspect": has_temporal,
                "query_length": len(query),
                "estimated_complexity": self._estimate_complexity(query)
            }
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {str(e)}")
            return {
                "embedding_vector": [],
                "word_count": len(query.split()),
                "has_question_word": False,
                "is_comparison": False,
                "has_temporal_aspect": False,
                "query_length": len(query),
                "estimated_complexity": "medium"
            }
    
    async def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from the query"""
        try:
            # Simple entity extraction (could be enhanced with NER models)
            words = query.split()
            
            # Detect potential entities
            capitalized_words = [word for word in words if word[0].isupper() and len(word) > 1]
            numbers = [word for word in words if word.isdigit()]
            
            # Detect dates (simple patterns)
            import re
            date_patterns = [
                r'\d{4}',  # Years
                r'\d{1,2}/\d{1,2}/\d{4}',  # Dates
                r'\d{1,2}-\d{1,2}-\d{4}'   # Dates
            ]
            
            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, query))
            
            return {
                "named_entities": capitalized_words,
                "numbers": numbers,
                "dates": dates,
                "total_entities": len(capitalized_words) + len(numbers) + len(dates)
            }
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {
                "named_entities": [],
                "numbers": [],
                "dates": [],
                "total_entities": 0
            }
    
    async def _select_best_expansion(
        self,
        original_query: str,
        expanded_queries: List[str]
    ) -> str:
        """Select the best query expansion"""
        try:
            # Get embeddings for all queries
            original_embedding = await self.embedding_manager.embed_text(original_query)
            
            best_expansion = expanded_queries[0]  # Default to first
            best_score = 0.0
            
            for expansion in expanded_queries:
                # Simple scoring based on length and diversity
                length_score = min(len(expansion) / len(original_query), 2.0) / 2.0
                diversity_score = len(set(expansion.split()) - set(original_query.split())) / len(expansion.split())
                
                combined_score = (length_score + diversity_score) / 2.0
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_expansion = expansion
            
            return best_expansion
            
        except Exception as e:
            logger.error(f"Error selecting best expansion: {str(e)}")
            return expanded_queries[0] if expanded_queries else ""
    
    def _validate_intent(self, query: str, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance intent detection results"""
        # Add heuristic validation
        query_lower = query.lower()
        
        # Override based on simple patterns
        if "?" in query:
            if intent_result.get("primary_intent") not in ["information_seeking", "explanation"]:
                intent_result["primary_intent"] = "information_seeking"
                intent_result["confidence"] = min(intent_result.get("confidence", 0.5) + 0.2, 1.0)
        
        if any(word in query_lower for word in ["how to", "step by step", "guide"]):
            intent_result["primary_intent"] = "how_to"
            intent_result["confidence"] = min(intent_result.get("confidence", 0.5) + 0.3, 1.0)
        
        if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
            intent_result["primary_intent"] = "comparison"
            intent_result["confidence"] = min(intent_result.get("confidence", 0.5) + 0.3, 1.0)
        
        return intent_result
    
    def _validate_routing(
        self,
        routing_result: Dict[str, Any],
        available_domains: List[str]
    ) -> Dict[str, Any]:
        """Validate routing results"""
        primary_domain = routing_result.get("primary_domain", "general")
        
        # Ensure primary domain is valid
        if primary_domain not in available_domains:
            routing_result["primary_domain"] = "general"
        
        # Validate secondary domains
        secondary_domains = routing_result.get("secondary_domains", [])
        valid_secondary = [d for d in secondary_domains if d in available_domains]
        routing_result["secondary_domains"] = valid_secondary
        
        # Ensure confidence is in valid range
        confidence = routing_result.get("confidence", 0.5)
        routing_result["confidence"] = max(0.0, min(1.0, confidence))
        
        return routing_result
    
    def _estimate_complexity(self, query: str) -> str:
        """Estimate query complexity"""
        word_count = len(query.split())
        char_count = len(query)
        
        # Simple heuristics
        if word_count < 5 and char_count < 30:
            return "simple"
        elif word_count > 15 or char_count > 100:
            return "complex"
        else:
            return "medium"