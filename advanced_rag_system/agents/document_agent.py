"""
Document Agent - Handles intelligent document processing and analysis
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger

from ..utils.config import AgentConfig, LLMConfig
from .base_agent import BaseAgent
from ..processors.document_processor import DocumentProcessor

class DocumentAgent(BaseAgent):
    """
    Document Agent responsible for:
    - Intelligent document analysis and understanding
    - Optimal chunking strategies
    - Metadata extraction and enrichment
    - Quality assessment of processed documents
    - Document classification and tagging
    """
    
    def __init__(
        self,
        agent_config: AgentConfig,
        llm_config: LLMConfig,
        document_processor: DocumentProcessor
    ):
        super().__init__(agent_config, llm_config)
        self.document_processor = document_processor
        
        # Document analysis templates
        self.analysis_template = """
        Analyze this document and provide structured information:
        
        Document Title: {title}
        Document Type: {doc_type}
        Content Preview: {content_preview}
        
        Provide:
        1. Main topics and themes
        2. Document quality assessment (1-10)
        3. Recommended chunk size
        4. Key metadata to extract
        5. Suggested tags/categories
        
        Format as JSON:
        {{
            "topics": [...],
            "quality_score": 0,
            "recommended_chunk_size": 0,
            "key_metadata": [...],
            "suggested_tags": [...],
            "document_type": "...",
            "complexity_level": "..."
        }}
        """
        
        # Metadata extraction template
        self.metadata_template = """
        Extract relevant metadata from this document:
        
        Title: {title}
        Content: {content}
        File Info: {file_info}
        
        Extract:
        1. Author information
        2. Creation/modification dates
        3. Subject matter/domain
        4. Key concepts
        5. Document structure
        6. Language and style
        
        Format as JSON:
        {{
            "author": "...",
            "created_date": "...",
            "domain": "...",
            "key_concepts": [...],
            "structure": "...",
            "language": "...",
            "style": "..."
        }}
        """
        
        # Quality assessment template
        self.quality_template = """
        Assess the quality of this document for RAG purposes:
        
        Content: {content}
        Length: {length}
        Structure: {structure}
        
        Evaluate:
        1. Information density (1-10)
        2. Clarity and readability (1-10)
        3. Factual accuracy indicators (1-10)
        4. Completeness (1-10)
        5. Relevance for knowledge base (1-10)
        
        Format as JSON:
        {{
            "information_density": 0,
            "clarity": 0,
            "accuracy_indicators": 0,
            "completeness": 0,
            "relevance": 0,
            "overall_score": 0,
            "recommendations": [...]
        }}
        """
    
    async def process(
        self,
        document_path: str,
        metadata: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Main processing method"""
        return await self.process_document(document_path, metadata, options)
    
    async def process_document(
        self,
        document_path: str,
        metadata: Dict[str, Any],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a document with intelligent analysis"""
        try:
            document_id = str(uuid.uuid4())
            logger.info(f"Processing document: {document_path} (ID: {document_id})")
            
            # Step 1: Basic document processing
            processed_doc = await self.document_processor.process_document(
                document_path, metadata
            )
            
            # Step 2: Intelligent analysis
            analysis_result = await self.analyze_document(
                processed_doc, document_path
            )
            
            # Step 3: Optimize chunking based on analysis
            optimized_chunks = await self.optimize_chunking(
                processed_doc, analysis_result
            )
            
            # Step 4: Extract and enrich metadata
            enriched_metadata = await self.extract_metadata(
                processed_doc, analysis_result, metadata
            )
            
            # Step 5: Quality assessment
            quality_assessment = await self.assess_quality(
                processed_doc, analysis_result
            )
            
            # Step 6: Generate embeddings and store
            embeddings_created = await self._create_and_store_embeddings(
                optimized_chunks, enriched_metadata, document_id
            )
            
            return {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(optimized_chunks),
                "embeddings_created": embeddings_created,
                "metadata": {
                    "original_metadata": metadata,
                    "enriched_metadata": enriched_metadata,
                    "analysis": analysis_result,
                    "quality_assessment": quality_assessment,
                    "processing_options": options
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}")
            return {
                "success": False,
                "document_id": "",
                "chunks_created": 0,
                "embeddings_created": 0,
                "metadata": {"error": str(e)}
            }
    
    async def analyze_document(
        self,
        processed_doc: Dict[str, Any],
        document_path: str
    ) -> Dict[str, Any]:
        """Perform intelligent document analysis"""
        try:
            content = processed_doc.get("content", "")
            title = processed_doc.get("metadata", {}).get("title", Path(document_path).name)
            doc_type = processed_doc.get("metadata", {}).get("type", "unknown")
            
            # Get content preview (first 1000 characters)
            content_preview = content[:1000] if content else ""
            
            prompt = self.analysis_template.format(
                title=title,
                doc_type=doc_type,
                content_preview=content_preview
            )
            
            response = await self.llm.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=600
            )
            
            import json
            analysis = json.loads(response.strip())
            
            # Add computed metrics
            analysis.update({
                "content_length": len(content),
                "word_count": len(content.split()) if content else 0,
                "estimated_reading_time": len(content.split()) / 200 if content else 0,  # 200 WPM
                "document_path": document_path
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            return {
                "topics": [],
                "quality_score": 5,
                "recommended_chunk_size": 1000,
                "key_metadata": [],
                "suggested_tags": [],
                "document_type": "unknown",
                "complexity_level": "medium",
                "content_length": 0,
                "word_count": 0,
                "estimated_reading_time": 0,
                "document_path": document_path
            }
    
    async def optimize_chunking(
        self,
        processed_doc: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Optimize chunking strategy based on document analysis"""
        try:
            content = processed_doc.get("content", "")
            recommended_chunk_size = analysis.get("recommended_chunk_size", 1000)
            
            # Adjust chunk size based on document characteristics
            if analysis.get("complexity_level") == "complex":
                chunk_size = min(recommended_chunk_size, 800)  # Smaller chunks for complex content
            elif analysis.get("complexity_level") == "simple":
                chunk_size = max(recommended_chunk_size, 1200)  # Larger chunks for simple content
            else:
                chunk_size = recommended_chunk_size
            
            # Use document processor with optimized settings
            chunks = await self.document_processor.chunk_text(
                content,
                chunk_size=chunk_size,
                overlap=int(chunk_size * 0.2),  # 20% overlap
                preserve_structure=True
            )
            
            # Enhance chunks with analysis metadata
            enhanced_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = {
                    "content": chunk,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "word_count": len(chunk.split()),
                    "analysis_metadata": {
                        "document_topics": analysis.get("topics", []),
                        "quality_score": analysis.get("quality_score", 5),
                        "complexity_level": analysis.get("complexity_level", "medium")
                    }
                }
                enhanced_chunks.append(enhanced_chunk)
            
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Error optimizing chunking: {str(e)}")
            # Fallback to simple chunking
            content = processed_doc.get("content", "")
            simple_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
            return [{"content": chunk, "chunk_id": i} for i, chunk in enumerate(simple_chunks)]
    
    async def extract_metadata(
        self,
        processed_doc: Dict[str, Any],
        analysis: Dict[str, Any],
        original_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and enrich document metadata"""
        try:
            content = processed_doc.get("content", "")[:2000]  # First 2000 chars for analysis
            title = processed_doc.get("metadata", {}).get("title", "")
            file_info = processed_doc.get("metadata", {})
            
            prompt = self.metadata_template.format(
                title=title,
                content=content,
                file_info=file_info
            )
            
            response = await self.llm.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            import json
            extracted_metadata = json.loads(response.strip())
            
            # Combine with original metadata and analysis
            enriched_metadata = {
                **original_metadata,
                **extracted_metadata,
                "analysis_derived": {
                    "topics": analysis.get("topics", []),
                    "suggested_tags": analysis.get("suggested_tags", []),
                    "quality_score": analysis.get("quality_score", 5),
                    "complexity_level": analysis.get("complexity_level", "medium")
                },
                "processing_timestamp": asyncio.get_event_loop().time(),
                "processor_version": "1.0.0"
            }
            
            return enriched_metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            return {
                **original_metadata,
                "extraction_error": str(e),
                "processing_timestamp": asyncio.get_event_loop().time()
            }
    
    async def assess_quality(
        self,
        processed_doc: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess document quality for RAG purposes"""
        try:
            content = processed_doc.get("content", "")
            content_sample = content[:1500] if content else ""  # Sample for analysis
            
            structure_info = {
                "has_headings": bool(processed_doc.get("structure", {}).get("headings")),
                "has_paragraphs": "\n\n" in content,
                "estimated_sections": content.count("\n\n") + 1
            }
            
            prompt = self.quality_template.format(
                content=content_sample,
                length=len(content),
                structure=structure_info
            )
            
            response = await self.llm.generate_response(
                prompt,
                temperature=0.3,
                max_tokens=400
            )
            
            import json
            quality_assessment = json.loads(response.strip())
            
            # Add computed quality metrics
            quality_assessment.update({
                "content_metrics": {
                    "length": len(content),
                    "word_count": len(content.split()) if content else 0,
                    "avg_sentence_length": self._calculate_avg_sentence_length(content),
                    "readability_score": self._estimate_readability(content)
                },
                "structure_metrics": structure_info
            })
            
            return quality_assessment
            
        except Exception as e:
            logger.error(f"Error assessing quality: {str(e)}")
            return {
                "information_density": 5,
                "clarity": 5,
                "accuracy_indicators": 5,
                "completeness": 5,
                "relevance": 5,
                "overall_score": 5,
                "recommendations": ["Quality assessment failed"],
                "assessment_error": str(e)
            }
    
    async def _create_and_store_embeddings(
        self,
        chunks: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        document_id: str
    ) -> int:
        """Create embeddings and store in vector database"""
        try:
            # This would integrate with the vector store manager
            # For now, return the count of chunks that would be embedded
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            return 0
    
    def _calculate_avg_sentence_length(self, content: str) -> float:
        """Calculate average sentence length"""
        try:
            sentences = content.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            if not sentences:
                return 0.0
            
            total_words = sum(len(sentence.split()) for sentence in sentences)
            return total_words / len(sentences)
            
        except Exception:
            return 0.0
    
    def _estimate_readability(self, content: str) -> float:
        """Simple readability estimation"""
        try:
            words = content.split()
            sentences = content.split('.')
            
            if not words or not sentences:
                return 0.0
            
            avg_words_per_sentence = len(words) / len(sentences)
            
            # Simple readability score (lower is more readable)
            # Based on average words per sentence
            if avg_words_per_sentence < 15:
                return 8.0  # Easy
            elif avg_words_per_sentence < 25:
                return 6.0  # Medium
            else:
                return 4.0  # Difficult
                
        except Exception:
            return 5.0  # Default medium score