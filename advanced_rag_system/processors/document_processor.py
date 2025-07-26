"""
Document Processor - Handles multi-format document parsing and chunking
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

from ..utils.config import ProcessingConfig


@dataclass
class ProcessedDocument:
    """Processed document with chunks and metadata"""
    id: str
    content: str
    chunks: List[str]
    metadata: Dict[str, Any]
    file_path: str


class DocumentProcessor:
    """
    Processor for multi-format documents
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
    
    async def process_document(self, file_path: Path) -> ProcessedDocument:
        """
        Process a document and extract content with chunks
        
        Args:
            file_path: Path to the document
            
        Returns:
            ProcessedDocument with content and chunks
        """
        # Extract content based on file type
        content = await self._extract_content(file_path)
        
        # Extract metadata
        metadata = await self._extract_file_metadata(file_path)
        
        # Extract document structure
        structure = await self._extract_structure(content)
        
        # Create chunks
        chunks = await self._smart_chunking(content, structure)
        
        # Generate document ID
        doc_id = f"doc_{hash(str(file_path))}"
        
        return ProcessedDocument(
            id=doc_id,
            content=content,
            chunks=chunks,
            metadata={
                **metadata,
                "structure": structure,
                "chunk_count": len(chunks),
                "file_path": str(file_path)
            },
            file_path=str(file_path)
        )
    
    async def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        Chunk text into smaller pieces
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size * 0.7:  # At least 70% of chunk
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def _extract_content(self, file_path: Path) -> str:
        """Extract content from different file formats"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == ".pdf":
            return await self._extract_pdf_content(file_path)
        elif file_extension == ".docx":
            return await self._extract_docx_content(file_path)
        elif file_extension == ".txt":
            return await self._extract_txt_content(file_path)
        elif file_extension == ".md":
            return await self._extract_md_content(file_path)
        elif file_extension == ".html":
            return await self._extract_html_content(file_path)
        elif file_extension == ".json":
            return await self._extract_json_content(file_path)
        elif file_extension == ".xml":
            return await self._extract_xml_content(file_path)
        else:
            # Default to text extraction
            return await self._extract_txt_content(file_path)
    
    async def _extract_pdf_content(self, file_path: Path) -> str:
        """Extract content from PDF file"""
        try:
            import pypdf
            
            with open(file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                text = ""
                
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
                
        except ImportError:
            raise ImportError("PyPDF is not installed. Install with: pip install pypdf")
        except Exception as e:
            # Fallback to OCR if text extraction fails
            if self.config.enable_ocr:
                return await self._ocr_pdf(file_path)
            else:
                raise Exception(f"Failed to extract PDF content: {e}")
    
    async def _extract_docx_content(self, file_path: Path) -> str:
        """Extract content from DOCX file"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
            
        except ImportError:
            raise ImportError("python-docx is not installed. Install with: pip install python-docx")
    
    async def _extract_txt_content(self, file_path: Path) -> str:
        """Extract content from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
    
    async def _extract_md_content(self, file_path: Path) -> str:
        """Extract content from Markdown file"""
        content = await self._extract_txt_content(file_path)
        
        # Remove markdown formatting (simplified)
        content = re.sub(r'#+\s+', '', content)  # Remove headers
        content = re.sub(r'\*\*(.*?)\*\*', r'\1', content)  # Remove bold
        content = re.sub(r'\*(.*?)\*', r'\1', content)  # Remove italic
        content = re.sub(r'`(.*?)`', r'\1', content)  # Remove code
        
        return content
    
    async def _extract_html_content(self, file_path: Path) -> str:
        """Extract content from HTML file"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text
                
        except ImportError:
            raise ImportError("BeautifulSoup is not installed. Install with: pip install beautifulsoup4")
    
    async def _extract_json_content(self, file_path: Path) -> str:
        """Extract content from JSON file"""
        import json
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Convert JSON to readable text
            if isinstance(data, dict):
                return self._dict_to_text(data)
            elif isinstance(data, list):
                return '\n'.join(str(item) for item in data)
            else:
                return str(data)
    
    async def _extract_xml_content(self, file_path: Path) -> str:
        """Extract content from XML file"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'xml')
                return soup.get_text().strip()
                
        except ImportError:
            raise ImportError("BeautifulSoup is not installed. Install with: pip install beautifulsoup4")
    
    async def _extract_file_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract file metadata"""
        metadata = {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "file_size": file_path.stat().st_size if file_path.exists() else 0,
            "file_type": self._get_file_type(file_path.suffix.lower())
        }
        
        return metadata
    
    async def _extract_structure(self, content: str) -> Dict[str, Any]:
        """Extract document structure information"""
        lines = content.split('\n')
        
        structure = {
            "has_headings": False,
            "has_paragraphs": False,
            "has_lists": False,
            "line_count": len(lines),
            "word_count": len(content.split()),
            "sentence_count": len(content.split('.')),
            "paragraph_count": content.count('\n\n') + 1
        }
        
        # Detect headings (lines that are all caps or start with #)
        for line in lines:
            if line.strip().isupper() or line.strip().startswith('#'):
                structure["has_headings"] = True
                break
        
        # Detect paragraphs
        if content.count('\n\n') > 0:
            structure["has_paragraphs"] = True
        
        # Detect lists
        for line in lines:
            if line.strip().startswith(('-', '*', '1.', '2.')):
                structure["has_lists"] = True
                break
        
        return structure
    
    async def _smart_chunking(self, content: str, structure: Dict[str, Any]) -> List[str]:
        """Perform smart chunking based on document structure"""
        if structure.get("has_headings"):
            # Chunk by sections
            return await self._section_based_chunking(content)
        elif structure.get("has_paragraphs"):
            # Chunk by paragraphs
            return await self._paragraph_based_chunking(content)
        else:
            # Use sliding window chunking
            return await self._sliding_window_chunking(content)
    
    async def _section_based_chunking(self, content: str) -> List[str]:
        """Chunk content by sections/headings"""
        sections = re.split(r'\n(?=#+\s|\n[A-Z][^\n]*\n)', content)
        chunks = []
        
        for section in sections:
            if section.strip():
                section_chunks = await self.chunk_text(section.strip())
                chunks.extend(section_chunks)
        
        return chunks
    
    async def _paragraph_based_chunking(self, content: str) -> List[str]:
        """Chunk content by paragraphs"""
        paragraphs = content.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                paragraph_chunks = await self.chunk_text(paragraph.strip())
                chunks.extend(paragraph_chunks)
        
        return chunks
    
    async def _sliding_window_chunking(self, content: str) -> List[str]:
        """Use sliding window chunking"""
        return await self.chunk_text(content)
    
    def _get_file_type(self, extension: str) -> str:
        """Get file type based on extension"""
        type_map = {
            ".pdf": "pdf",
            ".docx": "word",
            ".txt": "text",
            ".md": "markdown",
            ".html": "web",
            ".json": "data",
            ".xml": "data"
        }
        return type_map.get(extension, "unknown")
    
    def _dict_to_text(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Convert dictionary to readable text"""
        text = ""
        for key, value in data.items():
            if isinstance(value, dict):
                text += f"{'  ' * indent}{key}:\n"
                text += self._dict_to_text(value, indent + 1)
            elif isinstance(value, list):
                text += f"{'  ' * indent}{key}:\n"
                for item in value:
                    if isinstance(item, dict):
                        text += self._dict_to_text(item, indent + 1)
                    else:
                        text += f"{'  ' * (indent + 1)}{item}\n"
            else:
                text += f"{'  ' * indent}{key}: {value}\n"
        return text
    
    async def _ocr_pdf(self, file_path: Path) -> str:
        """Extract text from PDF using OCR"""
        # Placeholder for OCR implementation
        # In production, use libraries like pytesseract or OCR APIs
        return f"OCR content from {file_path.name}" 