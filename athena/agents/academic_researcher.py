"""
Academic Research Agent for Athena

Specialized agent for conducting academic and scholarly research using
academic databases, journal articles, and research publications.
"""

import asyncio
import arxiv
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import re

from scholarly import scholarly
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from ..core.base_agent import BaseAgent, ResearchTask, ResearchResult


class AcademicResearchAgent(BaseAgent):
    """
    Academic Research Agent specializing in scholarly and academic research.
    
    Capabilities:
    - ArXiv paper search and analysis
    - Google Scholar integration
    - Academic database queries
    - Citation analysis and tracking
    - Research trend identification
    - Peer review and quality assessment
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            agent_id="academic_researcher",
            name="Academic Research Agent",
            description="Specialized agent for academic and scholarly research",
            capabilities=[
                "arxiv_search",
                "google_scholar",
                "citation_analysis",
                "paper_analysis",
                "research_trends",
                "quality_assessment"
            ],
            config=config
        )
        
        # Initialize LLM for paper analysis
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=openai_api_key
        ) if openai_api_key else None
        
        # Research configuration
        self.max_papers = config.get("max_papers", 10)
        self.max_abstract_length = config.get("max_abstract_length", 2000)
        self.min_citation_count = config.get("min_citation_count", 5)
        
        self.logger.info("Academic Research Agent initialized")
    
    def can_handle_task(self, task: ResearchTask) -> bool:
        """Determine if this agent can handle the given task"""
        academic_indicators = [
            "research", "study", "paper", "academic", "scholarly",
            "journal", "publication", "citation", "peer review",
            "theory", "methodology", "analysis", "experiment",
            "literature review", "meta-analysis"
        ]
        
        query_lower = task.query.lower()
        return any(indicator in query_lower for indicator in academic_indicators)
    
    async def process_task(self, task: ResearchTask) -> ResearchResult:
        """Process an academic research task"""
        self.logger.info(f"Processing academic research task: {task.query}")
        
        try:
            # Search academic sources
            arxiv_papers = await self._search_arxiv(task.query)
            scholar_papers = await self._search_google_scholar(task.query)
            
            # Combine and analyze papers
            all_papers = self._combine_and_deduplicate_papers(arxiv_papers, scholar_papers)
            
            # Analyze paper relevance and quality
            analyzed_papers = await self._analyze_papers(all_papers, task.query)
            
            # Generate academic research summary
            research_summary = await self._generate_academic_summary(
                task.query, analyzed_papers
            )
            
            # Extract sources (paper URLs/DOIs)
            sources = [paper.get("url", "") or paper.get("doi", "") for paper in analyzed_papers[:5]]
            sources = [s for s in sources if s]  # Remove empty sources
            
            # Calculate confidence based on paper quality and relevance
            confidence = self._calculate_academic_confidence(analyzed_papers)
            
            return ResearchResult(
                task_id=task.id,
                agent_id=self.agent_id,
                content=research_summary,
                sources=sources,
                confidence=confidence,
                metadata={
                    "arxiv_papers": len(arxiv_papers),
                    "scholar_papers": len(scholar_papers),
                    "total_analyzed": len(analyzed_papers),
                    "high_quality_papers": len([p for p in analyzed_papers if p.get("quality_score", 0) > 0.7]),
                    "processing_time": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in academic research: {e}")
            raise
    
    async def _search_arxiv(self, query: str) -> List[Dict[str, Any]]:
        """Search ArXiv for relevant papers"""
        papers = []
        
        try:
            # Search ArXiv
            search = arxiv.Search(
                query=query,
                max_results=self.max_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in search.results():
                papers.append({
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary[:self.max_abstract_length],
                    "published": paper.published.isoformat(),
                    "url": paper.entry_id,
                    "doi": paper.doi,
                    "categories": paper.categories,
                    "source": "arxiv"
                })
                
        except Exception as e:
            self.logger.warning(f"ArXiv search failed: {e}")
        
        return papers
    
    async def _search_google_scholar(self, query: str) -> List[Dict[str, Any]]:
        """Search Google Scholar for relevant papers"""
        papers = []
        
        try:
            # Search Google Scholar (limited to avoid rate limiting)
            search_query = scholarly.search_pubs(query)
            
            count = 0
            for paper in search_query:
                if count >= min(self.max_papers // 2, 5):  # Limit Scholar results
                    break
                
                try:
                    # Fill paper details
                    filled_paper = scholarly.fill(paper)
                    
                    papers.append({
                        "title": filled_paper.get("title", ""),
                        "authors": filled_paper.get("author", []),
                        "abstract": filled_paper.get("abstract", "")[:self.max_abstract_length],
                        "published": filled_paper.get("pub_year", ""),
                        "url": filled_paper.get("pub_url", ""),
                        "citations": filled_paper.get("num_citations", 0),
                        "venue": filled_paper.get("venue", ""),
                        "source": "google_scholar"
                    })
                    count += 1
                    
                except Exception as e:
                    self.logger.debug(f"Error processing Scholar paper: {e}")
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Google Scholar search failed: {e}")
        
        return papers
    
    def _combine_and_deduplicate_papers(
        self,
        arxiv_papers: List[Dict[str, Any]],
        scholar_papers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Combine papers from different sources and remove duplicates"""
        all_papers = []
        seen_titles = set()
        
        # Process all papers
        for paper_list in [arxiv_papers, scholar_papers]:
            for paper in paper_list:
                title = paper.get("title", "").lower().strip()
                
                # Simple deduplication based on title similarity
                if title and not any(self._titles_similar(title, seen) for seen in seen_titles):
                    seen_titles.add(title)
                    all_papers.append(paper)
        
        # Sort by relevance indicators (citations, recency)
        all_papers.sort(key=lambda x: (
            x.get("citations", 0),
            x.get("published", "")
        ), reverse=True)
        
        return all_papers[:self.max_papers]
    
    def _titles_similar(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles are similar (simple word overlap)"""
        words1 = set(title1.lower().split())
        words2 = set(title2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union > threshold
    
    async def _analyze_papers(
        self,
        papers: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Analyze papers for relevance and quality"""
        analyzed_papers = []
        
        for paper in papers:
            analysis = await self._analyze_single_paper(paper, query)
            paper.update(analysis)
            
            # Only include papers with minimum relevance
            if paper.get("relevance_score", 0) > 0.3:
                analyzed_papers.append(paper)
        
        # Sort by combined relevance and quality score
        analyzed_papers.sort(
            key=lambda x: x.get("relevance_score", 0) * x.get("quality_score", 0),
            reverse=True
        )
        
        return analyzed_papers
    
    async def _analyze_single_paper(
        self,
        paper: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """Analyze a single paper for relevance and quality"""
        if not self.llm:
            return self._basic_paper_analysis(paper, query)
        
        analysis_prompt = f"""
        Analyze this academic paper for relevance to the research query and overall quality.
        
        Research Query: {query}
        
        Paper Details:
        Title: {paper.get('title', '')}
        Authors: {', '.join(paper.get('authors', []))}
        Abstract: {paper.get('abstract', '')[:500]}...
        Published: {paper.get('published', '')}
        Citations: {paper.get('citations', 'N/A')}
        Venue: {paper.get('venue', 'N/A')}
        
        Provide:
        1. Relevance score (0.0 to 1.0) - how well this paper addresses the query
        2. Quality score (0.0 to 1.0) - overall academic quality and rigor
        3. Key contributions related to the query
        4. Methodology assessment
        
        Format as: RELEVANCE: X.X | QUALITY: X.X | CONTRIBUTIONS: contribution1; contribution2 | METHODOLOGY: assessment
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert academic reviewer. Assess papers for relevance and quality concisely."),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            analysis_text = response.generations[0][0].text
            
            # Parse the response
            relevance_match = re.search(r'RELEVANCE:\s*([0-9.]+)', analysis_text)
            relevance_score = float(relevance_match.group(1)) if relevance_match else 0.5
            
            quality_match = re.search(r'QUALITY:\s*([0-9.]+)', analysis_text)
            quality_score = float(quality_match.group(1)) if quality_match else 0.5
            
            contributions_match = re.search(r'CONTRIBUTIONS:\s*([^|]+)', analysis_text)
            contributions = contributions_match.group(1).split(';') if contributions_match else []
            contributions = [c.strip() for c in contributions if c.strip()]
            
            methodology_match = re.search(r'METHODOLOGY:\s*([^|]+)', analysis_text)
            methodology = methodology_match.group(1).strip() if methodology_match else "Not assessed"
            
            return {
                "relevance_score": relevance_score,
                "quality_score": quality_score,
                "key_contributions": contributions,
                "methodology_assessment": methodology
            }
            
        except Exception as e:
            self.logger.warning(f"Paper analysis failed: {e}")
            return self._basic_paper_analysis(paper, query)
    
    def _basic_paper_analysis(self, paper: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Basic paper analysis without LLM"""
        # Simple keyword matching for relevance
        query_words = set(query.lower().split())
        title_words = set(paper.get("title", "").lower().split())
        abstract_words = set(paper.get("abstract", "").lower().split())
        
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        abstract_overlap = len(query_words.intersection(abstract_words)) / len(query_words) if query_words else 0
        
        relevance_score = min((title_overlap * 0.7) + (abstract_overlap * 0.3), 1.0)
        
        # Simple quality assessment based on available metrics
        citations = paper.get("citations", 0)
        has_doi = bool(paper.get("doi"))
        has_venue = bool(paper.get("venue"))
        
        quality_indicators = [
            citations > self.min_citation_count,
            has_doi,
            has_venue,
            len(paper.get("abstract", "")) > 100
        ]
        
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        return {
            "relevance_score": relevance_score,
            "quality_score": quality_score,
            "key_contributions": [],
            "methodology_assessment": "Basic assessment only"
        }
    
    async def _generate_academic_summary(
        self,
        query: str,
        analyzed_papers: List[Dict[str, Any]]
    ) -> str:
        """Generate a comprehensive academic research summary"""
        if not self.llm:
            return self._generate_basic_academic_summary(query, analyzed_papers)
        
        # Prepare paper information for summary
        paper_summaries = []
        for paper in analyzed_papers[:5]:  # Top 5 papers
            paper_summaries.append({
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "contributions": paper.get("key_contributions", []),
                "methodology": paper.get("methodology_assessment", ""),
                "relevance": paper.get("relevance_score", 0),
                "quality": paper.get("quality_score", 0)
            })
        
        summary_prompt = f"""
        Create a comprehensive academic research summary based on the following scholarly papers.
        
        Research Query: {query}
        
        Analyzed Papers ({len(paper_summaries)} high-quality sources):
        
        {self._format_papers_for_summary(paper_summaries)}
        
        Provide an academic-style summary that includes:
        1. Literature Overview - key themes and trends
        2. Methodological Approaches - research methods used
        3. Key Findings and Contributions
        4. Theoretical Frameworks
        5. Research Gaps and Future Directions
        6. Critical Analysis and Synthesis
        
        Write in an academic tone with proper scholarly structure.
        """
        
        try:
            messages = [
                SystemMessage(content="You are an expert academic researcher. Create comprehensive, scholarly research summaries with proper academic structure and tone."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
            
        except Exception as e:
            self.logger.error(f"Academic summary generation failed: {e}")
            return self._generate_basic_academic_summary(query, analyzed_papers)
    
    def _format_papers_for_summary(self, paper_summaries: List[Dict[str, Any]]) -> str:
        """Format paper information for summary generation"""
        formatted = []
        for i, paper in enumerate(paper_summaries, 1):
            authors_str = ", ".join(paper.get("authors", [])[:3])  # First 3 authors
            if len(paper.get("authors", [])) > 3:
                authors_str += " et al."
            
            formatted.append(f"""
            Paper {i}: {paper.get('title', 'Untitled')}
            Authors: {authors_str}
            Relevance Score: {paper.get('relevance', 0):.2f}
            Quality Score: {paper.get('quality', 0):.2f}
            Key Contributions: {'; '.join(paper.get('contributions', []))}
            Methodology: {paper.get('methodology', 'Not specified')}
            """)
        return "\n".join(formatted)
    
    def _generate_basic_academic_summary(
        self,
        query: str,
        analyzed_papers: List[Dict[str, Any]]
    ) -> str:
        """Generate basic academic summary without LLM"""
        summary_parts = [f"Academic Research Summary for: {query}\n"]
        
        summary_parts.append(f"Literature Review: Analyzed {len(analyzed_papers)} scholarly papers.")
        
        if analyzed_papers:
            # Extract key information
            high_quality_papers = [p for p in analyzed_papers if p.get("quality_score", 0) > 0.7]
            recent_papers = [p for p in analyzed_papers if "2020" in str(p.get("published", ""))]
            
            summary_parts.append(f"High-quality papers: {len(high_quality_papers)}")
            summary_parts.append(f"Recent publications (2020+): {len(recent_papers)}")
            
            # Top papers
            summary_parts.append("\nKey Papers:")
            for i, paper in enumerate(analyzed_papers[:3], 1):
                authors = ", ".join(paper.get("authors", [])[:2])
                summary_parts.append(f"{i}. {paper.get('title', 'Untitled')} - {authors}")
            
            # Aggregate contributions
            all_contributions = []
            for paper in analyzed_papers:
                all_contributions.extend(paper.get("key_contributions", []))
            
            if all_contributions:
                summary_parts.append("\nKey Research Contributions:")
                for contrib in all_contributions[:5]:
                    summary_parts.append(f"• {contrib}")
        
        return "\n".join(summary_parts)
    
    def _calculate_academic_confidence(self, analyzed_papers: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for academic research"""
        if not analyzed_papers:
            return 0.0
        
        # Factors affecting confidence
        avg_relevance = sum(p.get("relevance_score", 0) for p in analyzed_papers) / len(analyzed_papers)
        avg_quality = sum(p.get("quality_score", 0) for p in analyzed_papers) / len(analyzed_papers)
        
        # Number of papers factor
        papers_factor = min(len(analyzed_papers) / 5.0, 1.0)
        
        # High-quality papers factor
        high_quality_count = len([p for p in analyzed_papers if p.get("quality_score", 0) > 0.7])
        quality_factor = min(high_quality_count / 3.0, 1.0)
        
        # Combined confidence score
        confidence = (avg_relevance * 0.3) + (avg_quality * 0.3) + (papers_factor * 0.2) + (quality_factor * 0.2)
        
        return min(confidence, 1.0)