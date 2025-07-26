"""
Web Research Agent for Athena

Specialized agent for conducting comprehensive web-based research using
multiple search engines, web scraping, and content analysis.
"""

import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from urllib.parse import urljoin, urlparse
import re

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from newspaper import Article
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..core.base_agent import BaseAgent, ResearchTask, ResearchResult


class WebResearchAgent(BaseAgent):
    """
    Web Research Agent specializing in internet-based information gathering.
    
    Capabilities:
    - Multi-engine web search (DuckDuckGo, Google, Bing)
    - Intelligent web scraping and content extraction
    - News article analysis and summarization
    - Social media and forum monitoring
    - Real-time information gathering
    - Source credibility assessment
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            agent_id="web_researcher",
            name="Web Research Agent",
            description="Specialized agent for comprehensive web-based research and information gathering",
            capabilities=[
                "web_search",
                "content_extraction",
                "news_analysis",
                "real_time_data",
                "source_verification",
                "trend_analysis"
            ],
            config=config
        )
        
        # Initialize LLM for content analysis
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=openai_api_key
        ) if openai_api_key else None
        
        # Search configuration
        self.max_search_results = config.get("max_search_results", 10)
        self.max_content_length = config.get("max_content_length", 5000)
        self.search_timeout = config.get("search_timeout", 30)
        
        # Content extraction settings
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        self.logger.info("Web Research Agent initialized")
    
    def can_handle_task(self, task: ResearchTask) -> bool:
        """Determine if this agent can handle the given task"""
        # Web researcher can handle most general research tasks
        task_indicators = [
            "search", "find", "information", "recent", "current",
            "news", "trends", "online", "website", "article"
        ]
        
        query_lower = task.query.lower()
        return any(indicator in query_lower for indicator in task_indicators) or \
               task.context.get("task_type") in ["primary_research", "sub_research", "general"]
    
    async def process_task(self, task: ResearchTask) -> ResearchResult:
        """Process a web research task"""
        self.logger.info(f"Processing web research task: {task.query}")
        
        try:
            # Perform multi-source web search
            search_results = await self._perform_web_search(task.query)
            
            # Extract and analyze content from top results
            content_analysis = await self._analyze_search_results(search_results, task.query)
            
            # Generate comprehensive research summary
            research_summary = await self._generate_research_summary(
                task.query, content_analysis, search_results
            )
            
            # Extract sources
            sources = [result.get("url", "") for result in search_results[:5]]
            
            # Calculate confidence based on result quality
            confidence = self._calculate_confidence(search_results, content_analysis)
            
            return ResearchResult(
                task_id=task.id,
                agent_id=self.agent_id,
                content=research_summary,
                sources=sources,
                confidence=confidence,
                metadata={
                    "search_results_count": len(search_results),
                    "content_sources": len(content_analysis),
                    "search_engines_used": ["duckduckgo"],
                    "processing_time": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in web research: {e}")
            raise
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform comprehensive web search using multiple sources"""
        all_results = []
        
        # DuckDuckGo search
        try:
            ddg_results = await self._search_duckduckgo(query)
            all_results.extend(ddg_results)
        except Exception as e:
            self.logger.warning(f"DuckDuckGo search failed: {e}")
        
        # Remove duplicates and limit results
        unique_results = self._deduplicate_results(all_results)
        return unique_results[:self.max_search_results]
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        results = []
        
        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=self.max_search_results))
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "source": "duckduckgo"
                    })
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
        
        return results
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate search results"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results
    
    async def _analyze_search_results(
        self,
        search_results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Extract and analyze content from search results"""
        content_analysis = []
        
        # Process top results concurrently
        tasks = []
        for result in search_results[:5]:  # Limit to top 5 for detailed analysis
            task = self._extract_content_from_url(result["url"], result)
            tasks.append(task)
        
        # Wait for all content extraction tasks
        extracted_contents = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, content in enumerate(extracted_contents):
            if isinstance(content, Exception):
                self.logger.warning(f"Content extraction failed for {search_results[i]['url']}: {content}")
                continue
            
            if content and content.get("content"):
                # Analyze relevance and quality
                analysis = await self._analyze_content_relevance(content["content"], query)
                content.update(analysis)
                content_analysis.append(content)
        
        return content_analysis
    
    async def _extract_content_from_url(
        self,
        url: str,
        search_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract content from a URL"""
        try:
            # Try newspaper3k for article extraction
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text:
                return {
                    "url": url,
                    "title": article.title or search_result.get("title", ""),
                    "content": article.text[:self.max_content_length],
                    "publish_date": article.publish_date.isoformat() if article.publish_date else None,
                    "authors": article.authors,
                    "extraction_method": "newspaper3k"
                }
        except Exception as e:
            self.logger.debug(f"Newspaper3k extraction failed for {url}: {e}")
        
        # Fallback to basic web scraping
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.search_timeout)) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Extract text content
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        return {
                            "url": url,
                            "title": search_result.get("title", ""),
                            "content": text[:self.max_content_length],
                            "extraction_method": "beautifulsoup"
                        }
        except Exception as e:
            self.logger.debug(f"Web scraping failed for {url}: {e}")
        
        return None
    
    async def _analyze_content_relevance(
        self,
        content: str,
        query: str
    ) -> Dict[str, Any]:
        """Analyze content relevance to the research query"""
        if not self.llm:
            return {"relevance_score": 0.5, "key_points": []}
        
        analysis_prompt = f"""
        Analyze the relevance of the following content to the research query.
        
        Query: {query}
        
        Content: {content[:1000]}...
        
        Provide:
        1. Relevance score (0.0 to 1.0)
        2. Key points that relate to the query
        3. Content quality assessment
        
        Format as: RELEVANCE: X.X | KEY_POINTS: point1; point2; point3 | QUALITY: assessment
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert content analyst. Assess content relevance and quality concisely."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            analysis_text = response.generations[0][0].text
            
            # Parse the response
            relevance_match = re.search(r'RELEVANCE:\s*([0-9.]+)', analysis_text)
            relevance_score = float(relevance_match.group(1)) if relevance_match else 0.5
            
            key_points_match = re.search(r'KEY_POINTS:\s*([^|]+)', analysis_text)
            key_points = key_points_match.group(1).split(';') if key_points_match else []
            key_points = [point.strip() for point in key_points if point.strip()]
            
            quality_match = re.search(r'QUALITY:\s*([^|]+)', analysis_text)
            quality = quality_match.group(1).strip() if quality_match else "Unknown"
            
            return {
                "relevance_score": relevance_score,
                "key_points": key_points,
                "quality_assessment": quality
            }
            
        except Exception as e:
            self.logger.warning(f"Content analysis failed: {e}")
            return {"relevance_score": 0.5, "key_points": [], "quality_assessment": "Analysis failed"}
    
    async def _generate_research_summary(
        self,
        query: str,
        content_analysis: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Generate a comprehensive research summary"""
        if not self.llm:
            # Fallback summary without LLM
            return self._generate_basic_summary(query, content_analysis, search_results)
        
        # Prepare content for summary
        relevant_content = []
        for analysis in content_analysis:
            if analysis.get("relevance_score", 0) > 0.3:
                relevant_content.append({
                    "title": analysis.get("title", ""),
                    "content": analysis.get("content", "")[:500],
                    "key_points": analysis.get("key_points", [])
                })
        
        summary_prompt = f"""
        Create a comprehensive research summary based on the following web research findings.
        
        Research Query: {query}
        
        Findings from {len(relevant_content)} relevant sources:
        
        {self._format_content_for_summary(relevant_content)}
        
        Provide a well-structured summary that includes:
        1. Overview of findings
        2. Key insights and facts
        3. Different perspectives or viewpoints
        4. Current trends or developments
        5. Implications and conclusions
        
        Make the summary informative, accurate, and well-organized.
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert research analyst. Create comprehensive, well-structured research summaries."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return self._generate_basic_summary(query, content_analysis, search_results)
    
    def _format_content_for_summary(self, content_analysis: List[Dict[str, Any]]) -> str:
        """Format content analysis for summary generation"""
        formatted = []
        for i, content in enumerate(content_analysis, 1):
            formatted.append(f"""
            Source {i}: {content.get('title', 'Untitled')}
            Key Points: {'; '.join(content.get('key_points', []))}
            Content: {content.get('content', '')[:300]}...
            """)
        return "\n".join(formatted)
    
    def _generate_basic_summary(
        self,
        query: str,
        content_analysis: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> str:
        """Generate basic summary without LLM"""
        summary_parts = [f"Web Research Summary for: {query}\n"]
        
        summary_parts.append(f"Found {len(search_results)} search results from web sources.")
        
        if content_analysis:
            summary_parts.append(f"Analyzed {len(content_analysis)} sources in detail.")
            
            # Extract key points
            all_key_points = []
            for analysis in content_analysis:
                all_key_points.extend(analysis.get("key_points", []))
            
            if all_key_points:
                summary_parts.append("\nKey Findings:")
                for point in all_key_points[:10]:  # Limit to top 10
                    summary_parts.append(f"• {point}")
        
        return "\n".join(summary_parts)
    
    def _calculate_confidence(
        self,
        search_results: List[Dict[str, Any]],
        content_analysis: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for the research"""
        if not search_results:
            return 0.0
        
        # Base confidence on number of results and content quality
        results_score = min(len(search_results) / 10.0, 1.0)  # More results = higher confidence
        
        if content_analysis:
            # Average relevance score from content analysis
            relevance_scores = [c.get("relevance_score", 0.5) for c in content_analysis]
            content_score = sum(relevance_scores) / len(relevance_scores)
        else:
            content_score = 0.5
        
        # Combine scores
        confidence = (results_score * 0.4) + (content_score * 0.6)
        return min(confidence, 1.0)