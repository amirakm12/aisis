"""
Synthesis Agent for Athena

Specialized agent for synthesizing research findings from multiple sources
and creating comprehensive, coherent research summaries and insights.
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import re

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import TokenTextSplitter

from ..core.base_agent import BaseAgent, ResearchTask, ResearchResult


class SynthesisAgent(BaseAgent):
    """
    Synthesis Agent specializing in combining and synthesizing research findings.

    Capabilities:
    - Multi-source information synthesis
    - Coherent narrative construction
    - Insight extraction and analysis
    - Contradiction resolution
    - Evidence quality assessment
    - Comprehensive report generation
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            agent_id="synthesis_agent",
            name="Synthesis Agent",
            description="Specialized agent for synthesizing research findings from multiple sources",
            capabilities=[
                "multi_source_synthesis",
                "narrative_construction",
                "insight_extraction",
                "contradiction_resolution",
                "evidence_assessment",
                "report_generation"
            ],
            config=config
        )

        # Initialize LLM for synthesis
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",  # Use GPT-4 for better synthesis
            temperature=0.2,  # Slightly higher for creative synthesis
            openai_api_key=openai_api_key
        ) if openai_api_key else None

        # Synthesis configuration
        self.max_input_length = config.get("max_input_length", 15000)
        self.synthesis_depth = config.get("synthesis_depth", "comprehensive")
        self.include_contradictions = config.get("include_contradictions", True)

        # Text processing
        self.text_splitter = TokenTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )

        self.logger.info("Synthesis Agent initialized")

    def can_handle_task(self, task: ResearchTask) -> bool:
        """Determine if this agent can handle the given task"""
        synthesis_indicators = [
            "synthesize", "combine", "summary", "overview", "analysis",
            "integrate", "consolidate", "comprehensive", "report",
            "findings", "conclusions", "insights", "synthesis"
        ]

        query_lower = task.query.lower()
        has_synthesis_keywords = any(indicator in query_lower for indicator in synthesis_indicators)

        # Also handle tasks that explicitly request synthesis or have multiple sources
        has_multiple_sources = len(task.context.get("sources", [])) > 1
        is_synthesis_task = task.context.get("task_type") == "synthesis"

        return has_synthesis_keywords or has_multiple_sources or is_synthesis_task

    async def process_task(self, task: ResearchTask) -> ResearchResult:
        """Process a synthesis task"""
        self.logger.info(f"Processing synthesis task: {task.query}")

        try:
            # Extract research findings from context
            research_findings = self._extract_research_findings(task.context)

            # Analyze and structure the findings
            structured_findings = await self._structure_findings(research_findings, task.query)

            # Identify key themes and patterns
            themes_and_patterns = await self._identify_themes_and_patterns(structured_findings, task.query)

            # Resolve contradictions and conflicts
            contradiction_analysis = await self._analyze_contradictions(structured_findings)

            # Generate comprehensive synthesis
            synthesis_content = await self._generate_comprehensive_synthesis(
                task.query, structured_findings, themes_and_patterns, contradiction_analysis
            )

            # Extract evidence sources
            evidence_sources = self._extract_evidence_sources(research_findings)

            # Calculate synthesis confidence
            confidence = self._calculate_synthesis_confidence(
                research_findings, structured_findings, contradiction_analysis
            )

            return ResearchResult(
                task_id=task.id,
                agent_id=self.agent_id,
                content=synthesis_content,
                sources=evidence_sources,
                confidence=confidence,
                metadata={
                    "source_count": len(research_findings),
                    "themes_identified": len(themes_and_patterns.get("themes", [])),
                    "contradictions_found": len(contradiction_analysis.get("contradictions", [])),
                    "synthesis_depth": self.synthesis_depth,
                    "processing_time": datetime.now().isoformat()
                }
            )

        except Exception as e:
            self.logger.error(f"Error in synthesis: {e}")
            raise

    def _extract_research_findings(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract research findings from task context"""
        findings = []

        # Extract from research results if available
        if "research_results" in context:
            for agent_id, result in context["research_results"].items():
                if isinstance(result, dict):
                    findings.append({
                        "agent_id": agent_id,
                        "content": result.get("content", ""),
                        "sources": result.get("sources", []),
                        "confidence": result.get("confidence", 0.5),
                        "metadata": result.get("metadata", {})
                    })

        # Extract from direct sources if available
        if "sources" in context:
            for i, source in enumerate(context["sources"]):
                findings.append({
                    "agent_id": f"source_{i}",
                    "content": source.get("content", str(source)),
                    "sources": [source.get("url", "")],
                    "confidence": source.get("confidence", 0.5),
                    "metadata": {"source_type": "direct"}
                })

        # If no structured findings, try to extract from raw content
        if not findings and "content" in context:
            findings.append({
                "agent_id": "raw_content",
                "content": context["content"],
                "sources": [],
                "confidence": 0.5,
                "metadata": {"source_type": "raw"}
            })

        return findings

    async def _structure_findings(
        self,
        findings: List[Dict[str, Any]],
        query: str
    ) -> Dict[str, Any]:
        """Structure and organize research findings"""
        if not self.llm:
            return self._basic_structure_findings(findings, query)

        # Prepare findings for analysis
        findings_text = self._format_findings_for_analysis(findings)

        structure_prompt = f"""
        Analyze and structure the following research findings for the query: "{query}"

        Research Findings:
        {findings_text}

        Structure the findings by:
        1. Categorizing information by theme/topic
        2. Identifying key facts and claims
        3. Noting evidence quality and source reliability
        4. Highlighting important insights and conclusions
        5. Identifying areas of agreement and disagreement

        Provide a structured analysis that organizes the information logically.
        """

        try:
            messages = [
                SystemMessage(content="You are an expert research synthesizer. Structure and organize research findings clearly and logically."),
                HumanMessage(content=structure_prompt)
            ]

            response = await self.llm.agenerate([messages])
            structure_text = response.generations[0][0].text

            return {
                "structured_analysis": structure_text,
                "findings_count": len(findings),
                "total_sources": sum(len(f.get("sources", [])) for f in findings),
                "avg_confidence": sum(f.get("confidence", 0) for f in findings) / len(findings) if findings else 0
            }

        except Exception as e:
            self.logger.warning(f"Structured analysis failed: {e}")
            return self._basic_structure_findings(findings, query)

    def _basic_structure_findings(self, findings: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Basic structuring without LLM"""
        return {
            "structured_analysis": f"Analysis of {len(findings)} research findings for: {query}",
            "findings_count": len(findings),
            "total_sources": sum(len(f.get("sources", [])) for f in findings),
            "avg_confidence": sum(f.get("confidence", 0) for f in findings) / len(findings) if findings else 0
        }

    async def _identify_themes_and_patterns(
        self,
        structured_findings: Dict[str, Any],
        query: str
    ) -> Dict[str, Any]:
        """Identify key themes and patterns in the research"""
        if not self.llm:
            return {"themes": [], "patterns": []}

        themes_prompt = f"""
        Identify key themes, patterns, and insights from the structured research findings.

        Query: {query}
        Structured Findings: {structured_findings.get('structured_analysis', '')}

        Identify:
        1. Major themes and topics that emerge
        2. Recurring patterns or trends
        3. Key insights and implications
        4. Relationships between different findings
        5. Novel or unexpected discoveries

        Provide a clear analysis of themes and patterns.
        """

        try:
            messages = [
                SystemMessage(content="You are an expert at identifying patterns and themes in research. Provide clear, insightful analysis."),
                HumanMessage(content=themes_prompt)
            ]

            response = await self.llm.agenerate([messages])
            themes_text = response.generations[0][0].text

            # Extract themes and patterns (simplified parsing)
            themes = self._extract_themes_from_text(themes_text)
            patterns = self._extract_patterns_from_text(themes_text)

            return {
                "themes": themes,
                "patterns": patterns,
                "analysis": themes_text
            }

        except Exception as e:
            self.logger.warning(f"Theme identification failed: {e}")
            return {"themes": [], "patterns": [], "analysis": "Theme analysis failed"}

    def _extract_themes_from_text(self, text: str) -> List[str]:
        """Extract themes from analysis text"""
        themes = []

        # Look for numbered lists or bullet points that might indicate themes
        theme_patterns = [
            r'(?:Theme|Topic)\s*\d+[:\-]\s*([^\n]+)',
            r'^\d+\.\s*([^\n]+)',
            r'[•▪▫]\s*([^\n]+)'
        ]

        for pattern in theme_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            themes.extend(matches)

        # Clean and deduplicate
        themes = [theme.strip() for theme in themes if len(theme.strip()) > 10]
        return list(set(themes))[:10]  # Limit to top 10

    def _extract_patterns_from_text(self, text: str) -> List[str]:
        """Extract patterns from analysis text"""
        patterns = []

        # Look for pattern-related keywords
        pattern_keywords = ['pattern', 'trend', 'recurring', 'consistent', 'relationship']

        sentences = text.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in pattern_keywords):
                clean_sentence = sentence.strip()
                if len(clean_sentence) > 20:
                    patterns.append(clean_sentence)

        return patterns[:5]  # Limit to top 5

    async def _analyze_contradictions(
        self,
        structured_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze contradictions and conflicts in the findings"""
        if not self.llm or not self.include_contradictions:
            return {"contradictions": [], "resolutions": []}

        contradiction_prompt = f"""
        Analyze the research findings for contradictions, conflicts, or disagreements.

        Findings: {structured_findings.get('structured_analysis', '')}

        Identify:
        1. Direct contradictions between sources
        2. Conflicting claims or evidence
        3. Disagreements in interpretation
        4. Potential reasons for conflicts
        5. Possible resolutions or explanations

        Provide analysis of contradictions and potential resolutions.
        """

        try:
            messages = [
                SystemMessage(content="You are an expert at identifying and resolving contradictions in research. Provide balanced, thoughtful analysis."),
                HumanMessage(content=contradiction_prompt)
            ]

            response = await self.llm.agenerate([messages])
            contradiction_text = response.generations[0][0].text

            return {
                "contradictions": self._extract_contradictions_from_text(contradiction_text),
                "resolutions": self._extract_resolutions_from_text(contradiction_text),
                "analysis": contradiction_text
            }

        except Exception as e:
            self.logger.warning(f"Contradiction analysis failed: {e}")
            return {"contradictions": [], "resolutions": [], "analysis": "Contradiction analysis failed"}

    def _extract_contradictions_from_text(self, text: str) -> List[str]:
        """Extract contradictions from analysis text"""
        contradictions = []

        # Look for contradiction indicators
        contradiction_patterns = [
            r'(?:contradiction|conflict|disagree)[:\-]\s*([^\n]+)',
            r'(?:however|but|whereas|while)[,\s]+([^\n\.]+)',
            r'(?:on the other hand|conversely|in contrast)[,\s]+([^\n\.]+)'
        ]

        for pattern in contradiction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            contradictions.extend(matches)

        return [c.strip() for c in contradictions if len(c.strip()) > 15][:5]

    def _extract_resolutions_from_text(self, text: str) -> List[str]:
        """Extract resolutions from analysis text"""
        resolutions = []

        # Look for resolution indicators
        resolution_patterns = [
            r'(?:resolution|explanation|reconcile)[:\-]\s*([^\n]+)',
            r'(?:this can be explained|this suggests|possible reason)[:\s]+([^\n\.]+)',
            r'(?:therefore|thus|consequently)[,\s]+([^\n\.]+)'
        ]

        for pattern in resolution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            resolutions.extend(matches)

        return [r.strip() for r in resolutions if len(r.strip()) > 15][:5]

    async def _generate_comprehensive_synthesis(
        self,
        query: str,
        structured_findings: Dict[str, Any],
        themes_patterns: Dict[str, Any],
        contradiction_analysis: Dict[str, Any]
    ) -> str:
        """Generate the final comprehensive synthesis"""
        if not self.llm:
            return self._generate_basic_synthesis(query, structured_findings, themes_patterns)

        synthesis_prompt = f"""
        Create a comprehensive research synthesis for the following query: "{query}"

        Structured Findings:
        {structured_findings.get('structured_analysis', '')}

        Key Themes and Patterns:
        {themes_patterns.get('analysis', '')}

        Contradiction Analysis:
        {contradiction_analysis.get('analysis', '')}

        Create a comprehensive synthesis that includes:

        1. EXECUTIVE SUMMARY
        - Brief overview of the research question and key findings
        - Main conclusions and implications

        2. KEY FINDINGS
        - Most important discoveries and insights
        - Supporting evidence and sources

        3. THEMATIC ANALYSIS
        - Major themes that emerged from the research
        - Patterns and trends identified

        4. CRITICAL ANALYSIS
        - Evaluation of evidence quality and reliability
        - Discussion of contradictions and their resolutions
        - Limitations and gaps in the research

        5. SYNTHESIS AND IMPLICATIONS
        - Integration of findings into coherent understanding
        - Broader implications and significance
        - Connections to existing knowledge

        6. CONCLUSIONS AND RECOMMENDATIONS
        - Clear conclusions based on the evidence
        - Recommendations for future research or action

        Write in a scholarly, comprehensive style appropriate for academic or professional use.
        """

        try:
            messages = [
                SystemMessage(content="You are an expert research synthesizer. Create comprehensive, well-structured syntheses that integrate multiple sources of information into coherent, insightful analyses."),
                HumanMessage(content=synthesis_prompt)
            ]

            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text

        except Exception as e:
            self.logger.error(f"Comprehensive synthesis generation failed: {e}")
            return self._generate_basic_synthesis(query, structured_findings, themes_patterns)

    def _generate_basic_synthesis(
        self,
        query: str,
        structured_findings: Dict[str, Any],
        themes_patterns: Dict[str, Any]
    ) -> str:
        """Generate basic synthesis without LLM"""
        synthesis_parts = [
            f"Research Synthesis: {query}",
            "",
            "EXECUTIVE SUMMARY",
            f"This synthesis combines findings from {structured_findings.get('findings_count', 0)} research sources.",
            f"Average confidence level: {structured_findings.get('avg_confidence', 0):.2f}",
            "",
            "KEY FINDINGS",
            "• Multiple sources were analyzed and synthesized",
            f"• {len(themes_patterns.get('themes', []))} major themes were identified",
            f"• {len(themes_patterns.get('patterns', []))} patterns were discovered",
            "",
            "THEMES IDENTIFIED"
        ]

        for theme in themes_patterns.get('themes', [])[:5]:
            synthesis_parts.append(f"• {theme}")

        synthesis_parts.extend([
            "",
            "CONCLUSIONS",
            "This synthesis provides an integrated view of the available research on the topic.",
            "Further analysis may be needed to resolve any remaining questions or contradictions."
        ])

        return "\n".join(synthesis_parts)

    def _format_findings_for_analysis(self, findings: List[Dict[str, Any]]) -> str:
        """Format findings for LLM analysis"""
        formatted = []

        for i, finding in enumerate(findings, 1):
            agent_id = finding.get("agent_id", f"source_{i}")
            content = finding.get("content", "")[:2000]  # Limit content length
            confidence = finding.get("confidence", 0.5)
            sources = finding.get("sources", [])

            formatted.append(f"""
            SOURCE {i} ({agent_id}):
            Confidence: {confidence:.2f}
            Sources: {', '.join(sources[:3])}
            Content: {content}
            ---
            """)

        return "\n".join(formatted)

    def _extract_evidence_sources(self, research_findings: List[Dict[str, Any]]) -> List[str]:
        """Extract all evidence sources from research findings"""
        all_sources = []
        
        for finding in research_findings:
            sources = finding.get("sources", [])
            all_sources.extend(sources)
        
        # Remove duplicates and empty sources
        unique_sources = list(set([s for s in all_sources if s and s.strip()]))
        return unique_sources[:10]  # Limit to top 10 sources

    def _calculate_synthesis_confidence(
        self,
        research_findings: List[Dict[str, Any]],
        structured_findings: Dict[str, Any],
        contradiction_analysis: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the synthesis"""
        if not research_findings:
            return 0.0

        confidence_factors = []

        # Source count factor
        source_count = len(research_findings)
        source_factor = min(source_count / 5.0, 1.0)  # More sources = higher confidence
        confidence_factors.append(source_factor)

        # Average source confidence
        avg_source_confidence = structured_findings.get("avg_confidence", 0.5)
        confidence_factors.append(avg_source_confidence)

        # Contradiction penalty
        contradiction_count = len(contradiction_analysis.get("contradictions", []))
        contradiction_penalty = max(0.0, 1.0 - (contradiction_count * 0.1))
        confidence_factors.append(contradiction_penalty)

        # Evidence diversity factor
        unique_agents = len(set(f.get("agent_id", "") for f in research_findings))
        diversity_factor = min(unique_agents / 3.0, 1.0)
        confidence_factors.append(diversity_factor)

        return sum(confidence_factors) / len(confidence_factors)