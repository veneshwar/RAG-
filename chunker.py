"""
Text chunker for streaming document processing
"""

import re
import logging
from typing import Dict, Any, List, Optional, Iterator
import math
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk"""
    chunk_id: str
    content: str
    start_index: int
    end_index: int
    metadata: Dict[str, Any]


class TextChunker:
    """Text chunker for breaking documents into manageable pieces"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: str = "fixed_size",  # "fixed_size", "semantic", "paragraph", "sentence"
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        separator: str = "\n\n"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.separator = separator
        
        # Compile regex patterns
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        self.paragraph_pattern = re.compile(r'\n\s*\n')
    
    async def chunk_text(self, text: str, document_id: str = None) -> List[Chunk]:
        """
        Chunk text according to the configured strategy
        """
        if not text:
            return []
        
        try:
            if self.chunk_strategy == "fixed_size":
                return await self._chunk_fixed_size(text, document_id)
            elif self.chunk_strategy == "semantic":
                return await self._chunk_semantic(text, document_id)
            elif self.chunk_strategy == "paragraph":
                return await self._chunk_paragraph(text, document_id)
            elif self.chunk_strategy == "sentence":
                return await self._chunk_sentence(text, document_id)
            else:
                logger.warning(f"Unknown chunk strategy: {self.chunk_strategy}, using fixed_size")
                return await self._chunk_fixed_size(text, document_id)
                
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return []
    
    async def _chunk_fixed_size(self, text: str, document_id: str = None) -> List[Chunk]:
        """
        Chunk text into fixed-size chunks with overlap
        """
        chunks = []
        text_length = len(text)
        
        start = 0
        chunk_index = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # If not the last chunk and not at end of text, try to break at word boundary
            if end < text_length:
                # Find last space before end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            # Create chunk
            chunk_content = text[start:end].strip()
            if len(chunk_content) >= self.min_chunk_size:
                chunk = Chunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}",
                    content=chunk_content,
                    start_index=start,
                    end_index=end,
                    metadata={
                        "chunk_index": chunk_index,
                        "chunk_strategy": "fixed_size",
                        "original_length": len(chunk_content),
                        "document_id": document_id
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Calculate next start position with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
    
    async def _chunk_semantic(self, text: str, document_id: str = None) -> List[Chunk]:
        """
        Chunk text based on semantic boundaries (simplified implementation)
        """
        # Split by paragraphs first
        paragraphs = self.paragraph_pattern.split(text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph exceeds max chunk size, create a new chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = Chunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}",
                        content=current_chunk.strip(),
                        start_index=0,  # Would need proper indexing
                        end_index=len(current_chunk),
                        metadata={
                            "chunk_index": chunk_index,
                            "chunk_strategy": "semantic",
                            "original_length": len(current_chunk),
                            "document_id": document_id,
                            "paragraph_count": current_chunk.count('\n\n') + 1
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with overlap from previous
                if self.chunk_overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap:]
                    current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk = Chunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}",
                content=current_chunk.strip(),
                start_index=0,
                end_index=len(current_chunk),
                metadata={
                    "chunk_index": chunk_index,
                    "chunk_strategy": "semantic",
                    "original_length": len(current_chunk),
                    "document_id": document_id,
                    "paragraph_count": current_chunk.count('\n\n') + 1
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def _chunk_paragraph(self, text: str, document_id: str = None) -> List[Chunk]:
        """
        Chunk text by paragraphs
        """
        paragraphs = self.paragraph_pattern.split(text)
        chunks = []
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if len(paragraph) >= self.min_chunk_size:
                chunk = Chunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}",
                    content=paragraph,
                    start_index=0,  # Would need proper indexing
                    end_index=len(paragraph),
                    metadata={
                        "chunk_index": chunk_index,
                        "chunk_strategy": "paragraph",
                        "original_length": len(paragraph),
                        "document_id": document_id
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    async def _chunk_sentence(self, text: str, document_id: str = None) -> List[Chunk]:
        """
        Chunk text by sentences
        """
        sentences = self.sentence_pattern.split(text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence exceeds max chunk size, create a new chunk
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk = Chunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}",
                        content=current_chunk.strip(),
                        start_index=0,
                        end_index=len(current_chunk),
                        metadata={
                            "chunk_index": chunk_index,
                            "chunk_strategy": "sentence",
                            "original_length": len(current_chunk),
                            "document_id": document_id,
                            "sentence_count": len(self.sentence_pattern.findall(current_chunk)) + 1
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk = Chunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}" if document_id else f"chunk_{chunk_index}",
                content=current_chunk.strip(),
                start_index=0,
                end_index=len(current_chunk),
                metadata={
                    "chunk_index": chunk_index,
                    "chunk_strategy": "sentence",
                    "original_length": len(current_chunk),
                    "document_id": document_id,
                    "sentence_count": len(self.sentence_pattern.findall(current_chunk)) + 1
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    async def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a complete document
        """
        try:
            content = document.get("content", "")
            document_id = document.get("document_id", "unknown")
            
            if not content:
                logger.warning(f"No content to chunk for document {document_id}")
                return []
            
            # Get chunks
            chunks = await self.chunk_text(content, document_id)
            
            # Convert chunks to document format
            chunked_documents = []
            for chunk in chunks:
                chunk_doc = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.metadata["chunk_index"],
                    "metadata": {
                        **document.get("metadata", {}),
                        **chunk.metadata,
                        "parent_document_id": document_id,
                        "chunk_type": "text_chunk"
                    }
                }
                chunked_documents.append(chunk_doc)
            
            logger.info(f"Created {len(chunked_documents)} chunks for document {document_id}")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Error chunking document {document.get('document_id')}: {str(e)}")
            return []
    
    async def chunk_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents in batch
        """
        all_chunks = []
        
        for doc in documents:
            chunks = await self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    async def get_chunking_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get chunking statistics
        """
        stats = {
            "total_documents": len(documents),
            "total_chunks": 0,
            "avg_chunks_per_document": 0.0,
            "avg_chunk_size": 0.0,
            "min_chunk_size": float('inf'),
            "max_chunk_size": 0,
            "chunk_strategy": self.chunk_strategy
        }
        
        total_chunk_size = 0
        
        for doc in documents:
            chunks = await self.chunk_document(doc)
            stats["total_chunks"] += len(chunks)
            
            for chunk in chunks:
                chunk_size = len(chunk.get("content", ""))
                total_chunk_size += chunk_size
                stats["min_chunk_size"] = min(stats["min_chunk_size"], chunk_size)
                stats["max_chunk_size"] = max(stats["max_chunk_size"], chunk_size)
        
        if stats["total_chunks"] > 0:
            stats["avg_chunks_per_document"] = round(stats["total_chunks"] / len(documents), 2)
            stats["avg_chunk_size"] = round(total_chunk_size / stats["total_chunks"], 2)
        
        if stats["min_chunk_size"] == float('inf'):
            stats["min_chunk_size"] = 0
        
        return stats


class StreamingTextChunker(TextChunker):
    """Text chunker optimized for streaming data"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = ""
        self._chunks_buffer = []
    
    async def chunk_stream_chunk(self, chunk: str, document_id: str = None) -> List[Chunk]:
        """
        Process a streaming chunk and return any complete chunks
        """
        if not chunk:
            return []
        
        # Add chunk to buffer
        self._buffer += chunk
        
        # Check if buffer is large enough to create chunks
        if len(self._buffer) >= self.chunk_size:
            chunks = await self.chunk_text(self._buffer, document_id)
            
            # Keep last chunk for overlap with next stream chunk
            if chunks and self.chunk_overlap > 0:
                last_chunk = chunks[-1]
                overlap_start = max(0, len(last_chunk.content) - self.chunk_overlap)
                self._buffer = last_chunk.content[overlap_start:]
                chunks = chunks[:-1]  # Remove last chunk as it's kept in buffer
            else:
                self._buffer = ""
            
            self._chunks_buffer.extend(chunks)
            return chunks
        
        return []
    
    async def flush_chunks(self, document_id: str = None) -> List[Chunk]:
        """
        Flush remaining buffer and return final chunks
        """
        if self._buffer:
            final_chunks = await self.chunk_text(self._buffer, document_id)
            self._chunks_buffer.extend(final_chunks)
            self._buffer = ""
            return final_chunks
        
        return []
    
    def get_all_chunks(self) -> List[Chunk]:
        """Get all chunks processed so far"""
        return self._chunks_buffer.copy()
    
    def reset_buffer(self):
        """Reset the internal buffer"""
        self._buffer = ""
        self._chunks_buffer = []
