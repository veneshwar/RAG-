"""
Data normalizer for streaming text processing
"""

import re
import logging
from typing import Dict, Any, List, Optional
import unicodedata
from datetime import datetime


logger = logging.getLogger(__name__)


class TextNormalizer:
    """Text normalizer for standardizing streaming data"""
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        normalize_quotes: bool = True,
        normalize_dashes: bool = True,
        normalize_whitespace: bool = True,
        remove_extra_newlines: bool = True,
        normalize_numbers: bool = False,
        normalize_dates: bool = False,
        case_normalization: str = "none"  # "none", "lower", "upper", "title"
    ):
        self.normalize_unicode = normalize_unicode
        self.normalize_quotes = normalize_quotes
        self.normalize_dashes = normalize_dashes
        self.normalize_whitespace = normalize_whitespace
        self.remove_extra_newlines = remove_extra_newlines
        self.normalize_numbers = normalize_numbers
        self.normalize_dates = normalize_dates
        self.case_normalization = case_normalization
        
        # Compile regex patterns
        self.quote_pattern = re.compile(r'[""''`]')
        self.dash_pattern = re.compile(r'[–—−‑]')
        self.newline_pattern = re.compile(r'\n\s*\n')
        self.whitespace_pattern = re.compile(r'\s+')
        self.number_pattern = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b')
        self.date_pattern = re.compile(r'\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b')
    
    async def normalize_text(self, text: str) -> str:
        """
        Normalize text according to configured rules
        """
        if not text:
            return ""
        
        try:
            # Step 1: Unicode normalization
            if self.normalize_unicode:
                text = unicodedata.normalize('NFKC', text)
            
            # Step 2: Quote normalization
            if self.normalize_quotes:
                text = self.quote_pattern.sub('"', text)
            
            # Step 3: Dash normalization
            if self.normalize_dashes:
                text = self.dash_pattern.sub('-', text)
            
            # Step 4: Number normalization
            if self.normalize_numbers:
                text = await self._normalize_numbers(text)
            
            # Step 5: Date normalization
            if self.normalize_dates:
                text = await self._normalize_dates(text)
            
            # Step 6: Newline normalization
            if self.remove_extra_newlines:
                text = self.newline_pattern.sub('\n\n', text)
            
            # Step 7: Whitespace normalization
            if self.normalize_whitespace:
                text = self.whitespace_pattern.sub(' ', text).strip()
            
            # Step 8: Case normalization
            text = self._normalize_case(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error normalizing text: {str(e)}")
            return text
    
    async def _normalize_numbers(self, text: str) -> str:
        """Normalize number formats"""
        def replace_number(match):
            num_str = match.group()
            # Remove commas and convert to standard format
            clean_num = num_str.replace(',', '')
            try:
                # Try to format as float
                num = float(clean_num)
                return f"{num:.2f}".rstrip('0').rstrip('.') if '.' in f"{num:.2f}" else str(int(num))
            except ValueError:
                return num_str
        
        return self.number_pattern.sub(replace_number, text)
    
    async def _normalize_dates(self, text: str) -> str:
        """Normalize date formats to ISO format"""
        def replace_date(match):
            date_str = match.group()
            try:
                # Simple date normalization - could be enhanced with dateutil
                parts = re.split(r'[/-]', date_str)
                if len(parts) == 3:
                    # Assume YYYY-MM-DD format if year is 4 digits
                    if len(parts[0]) == 4:
                        return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
                    # Otherwise assume MM-DD-YYYY
                    else:
                        return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
            except:
                pass
            return date_str
        
        return self.date_pattern.sub(replace_date, text)
    
    def _normalize_case(self, text: str) -> str:
        """Normalize text case"""
        if self.case_normalization == "lower":
            return text.lower()
        elif self.case_normalization == "upper":
            return text.upper()
        elif self.case_normalization == "title":
            return text.title()
        else:
            return text
    
    async def normalize_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a document's content
        """
        try:
            content = document.get("content", "")
            if not content:
                return document
            
            # Normalize the content
            normalized_content = await self.normalize_text(content)
            
            # Update document
            normalized_document = document.copy()
            normalized_document["content"] = normalized_content
            normalized_document["original_content"] = content
            normalized_document["normalization_metadata"] = {
                "original_length": len(content),
                "normalized_length": len(normalized_content),
                "normalization_applied": True,
                "normalization_timestamp": str(datetime.utcnow())
            }
            
            return normalized_document
            
        except Exception as e:
            logger.error(f"Error normalizing document: {str(e)}")
            return document
    
    async def normalize_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize multiple documents in batch
        """
        normalized_docs = []
        
        for doc in documents:
            normalized_doc = await self.normalize_document(doc)
            normalized_docs.append(normalized_doc)
        
        return normalized_docs
    
    async def get_normalization_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get normalization statistics
        """
        stats = {
            "total_documents": len(documents),
            "normalized_documents": 0,
            "failed_normalization": 0,
            "total_original_length": 0,
            "total_normalized_length": 0,
            "average_length_change": 0.0
        }
        
        for doc in documents:
            try:
                normalized_doc = await self.normalize_document(doc)
                
                if "normalization_metadata" in normalized_doc:
                    stats["normalized_documents"] += 1
                    stats["total_original_length"] += normalized_doc["normalization_metadata"]["original_length"]
                    stats["total_normalized_length"] += normalized_doc["normalization_metadata"]["normalized_length"]
                else:
                    stats["failed_normalization"] += 1
                    
            except Exception as e:
                logger.error(f"Error in normalization stats: {str(e)}")
                stats["failed_normalization"] += 1
        
        # Calculate average change
        if stats["total_original_length"] > 0:
            change = (stats["total_normalized_length"] - stats["total_original_length"]) / stats["total_original_length"]
            stats["average_length_change"] = round(change * 100, 2)
        
        return stats


class StreamingTextNormalizer(TextNormalizer):
    """Text normalizer optimized for streaming data"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = ""
        self._buffer_size = 1000
    
    async def normalize_stream_chunk(self, chunk: str) -> str:
        """
        Normalize a streaming chunk of text
        """
        if not chunk:
            return ""
        
        # Add chunk to buffer
        self._buffer += chunk
        
        # Process if buffer is large enough
        if len(self._buffer) >= self._buffer_size:
            normalized_chunk = await self.normalize_text(self._buffer)
            self._buffer = ""
            return normalized_chunk
        
        return ""
    
    async def flush_buffer(self) -> str:
        """
        Normalize and flush remaining buffer
        """
        if self._buffer:
            normalized_chunk = await self.normalize_text(self._buffer)
            self._buffer = ""
            return normalized_chunk
        return ""
    
    def reset_buffer(self):
        """Reset the internal buffer"""
        self._buffer = ""


class DocumentNormalizer:
    """High-level document normalizer with multiple normalization stages"""
    
    def __init__(
        self,
        text_normalizer: TextNormalizer,
        normalize_metadata: bool = True,
        extract_entities: bool = False
    ):
        self.text_normalizer = text_normalizer
        self.normalize_metadata = normalize_metadata
        self.extract_entities = extract_entities
    
    async def normalize_document_pipeline(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run document through complete normalization pipeline
        """
        try:
            # Stage 1: Text normalization
            normalized_doc = await self.text_normalizer.normalize_document(document)
            
            # Stage 2: Metadata normalization
            if self.normalize_metadata:
                normalized_doc = await self._normalize_metadata(normalized_doc)
            
            # Stage 3: Entity extraction (if enabled)
            if self.extract_entities:
                entities = await self._extract_entities(normalized_doc)
                normalized_doc["entities"] = entities
            
            # Stage 4: Quality metrics
            metrics = await self._calculate_quality_metrics(normalized_doc)
            normalized_doc["quality_metrics"] = metrics
            
            return normalized_doc
            
        except Exception as e:
            logger.error(f"Error in document normalization pipeline: {str(e)}")
            document["normalization_error"] = str(e)
            return document
    
    async def _normalize_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize document metadata
        """
        metadata = document.get("metadata", {})
        
        # Normalize string fields in metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                metadata[key] = await self.text_normalizer.normalize_text(value)
        
        # Add normalization timestamp
        metadata["normalization_timestamp"] = str(datetime.utcnow())
        
        document["metadata"] = metadata
        return document
    
    async def _extract_entities(self, document: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Extract basic entities from document (simplified implementation)
        """
        content = document.get("content", "")
        entities = {
            "emails": [],
            "urls": [],
            "numbers": [],
            "dates": []
        }
        
        # Extract emails
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        entities["emails"] = email_pattern.findall(content)
        
        # Extract URLs
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        entities["urls"] = url_pattern.findall(content)
        
        # Extract numbers
        number_pattern = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b')
        entities["numbers"] = number_pattern.findall(content)
        
        # Extract dates
        date_pattern = re.compile(r'\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b')
        entities["dates"] = date_pattern.findall(content)
        
        return entities
    
    async def _calculate_quality_metrics(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate quality metrics for normalized document
        """
        content = document.get("content", "")
        
        metrics = {
            "char_count": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(content.split('.')),
            "avg_word_length": 0.0,
            "readability_score": 0.0,
            "has_structure": False
        }
        
        if metrics["word_count"] > 0:
            # Average word length
            total_chars = sum(len(word) for word in content.split())
            metrics["avg_word_length"] = round(total_chars / metrics["word_count"], 2)
        
        # Simple readability check
        if metrics["sentence_count"] > 0:
            avg_sentence_length = metrics["word_count"] / metrics["sentence_count"]
            metrics["readability_score"] = min(avg_sentence_length / 20, 1.0)  # Ideal ~20 words per sentence
        
        # Check for document structure
        structure_indicators = ["introduction", "conclusion", "summary", "chapter", "section"]
        metrics["has_structure"] = any(indicator in content.lower() for indicator in structure_indicators)
        
        return metrics
