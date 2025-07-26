"""
Document Agent - Handles document analysis and processing
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from .base_agent import BaseAgent


@dataclass
class DocumentAnalysis:
    """Document analysis results"""
    document_type: str
    complexity: str
    key_topics: List[str]
    summary: str
    metadata: Dict[str, Any]


@dataclass
class ChunkingStrategy:
    """Chunking strategy for document processing"""
    strategy: str
    chunk_size: int
    overlap: int
    preserve_structure: bool


class DocumentAgent(BaseAgent):
    """
    Agent responsible for document analysis and processing
    """
    
    def __init__(self, config):
        super().__init__(config)
    
    async def process_document(
        self,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentAnalysis:
        """
        Process and analyze a document
        
        Args:
            file_path: Path to the document
            metadata: Additional metadata
            
        Returns:
            DocumentAnalysis with results
        """
        # Analyze document content
        analysis = await self.analyze_document(file_path)
        
        # Optimize chunking strategy
        chunking = await self.optimize_chunking(analysis)
        
        # Extract metadata
        extracted_metadata = await self.extract_metadata(file_path, analysis)
        
        # Assess quality
        quality_score = await self.assess_quality(analysis)
        
        return DocumentAnalysis(
            document_type=analysis.get("type", "unknown"),
            complexity=analysis.get("complexity", "medium"),
            key_topics=analysis.get("topics", []),
            summary=analysis.get("summary", ""),
            metadata={
                "chunking_strategy": chunking,
                "quality_score": quality_score,
                "extracted_metadata": extracted_metadata,
                **(metadata or {})
            }
        )
    
    async def analyze_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Analyze document content and structure
        
        Args:
            file_path: Path to the document
            
        Returns:
            Analysis results
        """
        # Read document content (simplified)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            content = ""
        
        # Simple analysis
        analysis = {
            "type": self._detect_document_type(file_path),
            "complexity": self._assess_complexity(content),
            "topics": self._extract_topics(content),
            "summary": self._generate_summary(content),
            "word_count": len(content.split()),
            "structure": self._analyze_structure(content)
        }
        
        return analysis
    
    async def optimize_chunking(self, analysis: Dict[str, Any]) -> ChunkingStrategy:
        """
        Optimize chunking strategy based on document analysis
        
        Args:
            analysis: Document analysis results
            
        Returns:
            Optimized chunking strategy
        """
        doc_type = analysis.get("type", "text")
        complexity = analysis.get("complexity", "medium")
        word_count = analysis.get("word_count", 0)
        
        # Determine optimal chunk size based on document characteristics
        if complexity == "high":
            chunk_size = 800
            overlap = 150
        elif complexity == "low":
            chunk_size = 1200
            overlap = 100
        else:
            chunk_size = 1000
            overlap = 200
        
        # Adjust for document type
        if doc_type == "technical":
            chunk_size = min(chunk_size, 600)
        elif doc_type == "narrative":
            chunk_size = max(chunk_size, 1200)
        
        return ChunkingStrategy(
            strategy="adaptive",
            chunk_size=chunk_size,
            overlap=overlap,
            preserve_structure=True
        )
    
    async def extract_metadata(
        self,
        file_path: Path,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract metadata from document
        
        Args:
            file_path: Path to the document
            analysis: Document analysis results
            
        Returns:
            Extracted metadata
        """
        metadata = {
            "filename": file_path.name,
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "file_type": file_path.suffix.lower(),
            "word_count": analysis.get("word_count", 0),
            "document_type": analysis.get("type", "unknown"),
            "complexity": analysis.get("complexity", "medium")
        }
        
        # Add file-specific metadata
        if file_path.suffix.lower() == ".pdf":
            metadata["pdf_metadata"] = self._extract_pdf_metadata(file_path)
        elif file_path.suffix.lower() == ".docx":
            metadata["docx_metadata"] = self._extract_docx_metadata(file_path)
        
        return metadata
    
    async def assess_quality(self, analysis: Dict[str, Any]) -> float:
        """
        Assess document quality
        
        Args:
            analysis: Document analysis results
            
        Returns:
            Quality score (0-1)
        """
        score = 0.5  # Base score
        
        # Adjust based on various factors
        word_count = analysis.get("word_count", 0)
        if word_count > 100:
            score += 0.2
        elif word_count < 10:
            score -= 0.3
        
        # Complexity factor
        complexity = analysis.get("complexity", "medium")
        if complexity == "high":
            score += 0.1
        elif complexity == "low":
            score -= 0.1
        
        # Structure factor
        structure = analysis.get("structure", {})
        if structure.get("has_headings", False):
            score += 0.1
        if structure.get("has_paragraphs", False):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def process(self, input_data: Any) -> Any:
        """Main processing method"""
        if isinstance(input_data, Path):
            return await self.process_document(input_data)
        elif isinstance(input_data, dict):
            file_path = Path(input_data.get("file_path", ""))
            metadata = input_data.get("metadata")
            return await self.process_document(file_path, metadata)
        else:
            return await self.process_document(Path(str(input_data)))
    
    def validate_input(self, input_data: Any) -> Any:
        """Validate input data"""
        if isinstance(input_data, (str, Path)):
            path = Path(input_data)
            if not path.exists():
                raise ValueError(f"File does not exist: {path}")
        elif isinstance(input_data, dict):
            if "file_path" not in input_data:
                raise ValueError("Input must contain 'file_path' field")
            path = Path(input_data["file_path"])
            if not path.exists():
                raise ValueError(f"File does not exist: {path}")
        else:
            raise ValueError("Input must be file path or dictionary")
        return input_data
    
    def _detect_document_type(self, file_path: Path) -> str:
        """Detect document type based on content and extension"""
        extension = file_path.suffix.lower()
        
        type_map = {
            ".pdf": "pdf",
            ".docx": "word",
            ".txt": "text",
            ".md": "markdown",
            ".html": "web",
            ".json": "data",
            ".xml": "data"
        }
        
        return type_map.get(extension, "text")
    
    def _assess_complexity(self, content: str) -> str:
        """Assess document complexity"""
        words = content.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        if avg_word_length > 8 or len(words) > 1000:
            return "high"
        elif avg_word_length < 4 or len(words) < 100:
            return "low"
        else:
            return "medium"
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract key topics from content"""
        # Simple keyword extraction (in production, use NLP)
        words = content.lower().split()
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 5 most frequent words
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _generate_summary(self, content: str) -> str:
        """Generate document summary"""
        # Simple summary (in production, use summarization models)
        sentences = content.split('.')
        if len(sentences) > 3:
            return '. '.join(sentences[:3]) + '.'
        return content[:200] + "..." if len(content) > 200 else content
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure"""
        lines = content.split('\n')
        
        return {
            "has_headings": any(line.strip().isupper() for line in lines),
            "has_paragraphs": content.count('\n\n') > 0,
            "has_lists": any(line.strip().startswith(('-', '*', '1.', '2.')) for line in lines),
            "line_count": len(lines)
        }
    
    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata"""
        # Placeholder for PDF metadata extraction
        return {"pages": 0, "title": "", "author": ""}
    
    def _extract_docx_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract DOCX metadata"""
        # Placeholder for DOCX metadata extraction
        return {"pages": 0, "title": "", "author": ""}
    
    async def _create_and_store_embeddings(
        self,
        chunks: List[str],
        metadata: Dict[str, Any]
    ) -> List[str]:
        """Create and store embeddings for document chunks"""
        # This would integrate with the embedding manager
        # For now, return placeholder IDs
        return [f"chunk_{i}" for i in range(len(chunks))] 