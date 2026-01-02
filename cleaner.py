"""
Data cleaner for streaming text processing
"""

import re
import html
import logging
from typing import Dict, Any, List, Optional
import unicodedata


logger = logging.getLogger(__name__)


class TextCleaner:
    """Text cleaner for preprocessing streaming data"""
    
    def __init__(
        self,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        lowercase: bool = False,
        remove_special_chars: bool = False,
        min_length: int = 10,
        max_length: Optional[int] = None
    ):
        self.remove_html = remove_html
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.min_length = min_length
        self.max_length = max_length
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.special_chars_pattern = re.compile(r'[^\w\s]')
    
    async def clean_text(self, text: str) -> str:
        """
        Clean and preprocess text
        """
        if not text:
            return ""
        
        try:
            # Step 1: Decode HTML entities
            text = html.unescape(text)
            
            # Step 2: Remove HTML tags
            if self.remove_html:
                text = self.html_pattern.sub(' ', text)
            
            # Step 3: Remove URLs
            if self.remove_urls:
                text = self.url_pattern.sub(' ', text)
            
            # Step 4: Remove emails
            if self.remove_emails:
                text = self.email_pattern.sub(' ', text)
            
            # Step 5: Normalize Unicode
            text = unicodedata.normalize('NFKC', text)
            
            # Step 6: Remove special characters
            if self.remove_special_chars:
                text = self.special_chars_pattern.sub(' ', text)
            
            # Step 7: Normalize whitespace
            if self.normalize_whitespace:
                text = self.whitespace_pattern.sub(' ', text).strip()
            
            # Step 8: Convert to lowercase
            if self.lowercase:
                text = text.lower()
            
            # Step 9: Length validation
            if len(text) < self.min_length:
                logger.warning(f"Text too short after cleaning: {len(text)} chars")
                return ""
            
            if self.max_length and len(text) > self.max_length:
                text = text[:self.max_length]
                logger.warning(f"Text truncated to max length: {self.max_length}")
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text  # Return original text on error
    
    async def clean_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a document's content
        """
        try:
            content = document.get("content", "")
            if not content:
                return document
            
            # Clean the content
            cleaned_content = await self.clean_text(content)
            
            # Update document
            cleaned_document = document.copy()
            cleaned_document["content"] = cleaned_content
            cleaned_document["original_content"] = content
            cleaned_document["cleaning_metadata"] = {
                "original_length": len(content),
                "cleaned_length": len(cleaned_content),
                "cleaning_applied": True
            }
            
            return cleaned_document
            
        except Exception as e:
            logger.error(f"Error cleaning document: {str(e)}")
            return document
    
    async def clean_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean multiple documents in batch
        """
        cleaned_docs = []
        
        for doc in documents:
            cleaned_doc = await self.clean_document(doc)
            cleaned_docs.append(cleaned_doc)
        
        return cleaned_docs
    
    def is_valid_text(self, text: str) -> bool:
        """
        Check if text meets quality criteria
        """
        if not text:
            return False
        
        # Length check
        if len(text) < self.min_length:
            return False
        
        if self.max_length and len(text) > self.max_length:
            return False
        
        # Content quality checks
        # Check if text has meaningful content (not just punctuation/numbers)
        meaningful_chars = sum(1 for c in text if c.isalnum())
        if meaningful_chars < len(text) * 0.3:  # At least 30% alphanumeric
            return False
        
        # Check for repeated patterns
        words = text.split()
        if len(set(words)) / len(words) < 0.3 if words else 0:  # At least 30% unique words
            return False
        
        return True
    
    async def get_cleaning_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get cleaning statistics
        """
        stats = {
            "total_documents": len(documents),
            "cleaned_documents": 0,
            "failed_cleaning": 0,
            "total_original_length": 0,
            "total_cleaned_length": 0,
            "average_length_reduction": 0.0
        }
        
        for doc in documents:
            try:
                cleaned_doc = await self.clean_document(doc)
                
                if "cleaning_metadata" in cleaned_doc:
                    stats["cleaned_documents"] += 1
                    stats["total_original_length"] += cleaned_doc["cleaning_metadata"]["original_length"]
                    stats["total_cleaned_length"] += cleaned_doc["cleaning_metadata"]["cleaned_length"]
                else:
                    stats["failed_cleaning"] += 1
                    
            except Exception as e:
                logger.error(f"Error in cleaning stats: {str(e)}")
                stats["failed_cleaning"] += 1
        
        # Calculate average reduction
        if stats["total_original_length"] > 0:
            reduction = (stats["total_original_length"] - stats["total_cleaned_length"]) / stats["total_original_length"]
            stats["average_length_reduction"] = round(reduction * 100, 2)
        
        return stats


class StreamingTextCleaner(TextCleaner):
    """Text cleaner optimized for streaming data"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._buffer = ""
        self._buffer_size = 1000
    
    async def clean_stream_chunk(self, chunk: str) -> str:
        """
        Clean a streaming chunk of text
        """
        if not chunk:
            return ""
        
        # Add chunk to buffer
        self._buffer += chunk
        
        # Process if buffer is large enough
        if len(self._buffer) >= self._buffer_size:
            cleaned_chunk = await self.clean_text(self._buffer)
            self._buffer = ""
            return cleaned_chunk
        
        return ""
    
    async def flush_buffer(self) -> str:
        """
        Clean and flush remaining buffer
        """
        if self._buffer:
            cleaned_chunk = await self.clean_text(self._buffer)
            self._buffer = ""
            return cleaned_chunk
        return ""
    
    def reset_buffer(self):
        """Reset the internal buffer"""
        self._buffer = ""


class DocumentCleaner:
    """High-level document cleaner with multiple cleaning stages"""
    
    def __init__(
        self,
        text_cleaner: TextCleaner,
        remove_duplicates: bool = True,
        deduplication_threshold: float = 0.9
    ):
        self.text_cleaner = text_cleaner
        self.remove_duplicates = remove_duplicates
        self.deduplication_threshold = deduplication_threshold
        self._seen_documents = set()
    
    async def clean_document_pipeline(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run document through complete cleaning pipeline
        """
        try:
            # Stage 1: Text cleaning
            cleaned_doc = await self.text_cleaner.clean_document(document)
            
            # Stage 2: Duplicate detection
            if self.remove_duplicates:
                if await self._is_duplicate(cleaned_doc):
                    logger.info(f"Duplicate document detected: {cleaned_doc.get('document_id')}")
                    cleaned_doc["is_duplicate"] = True
                    return cleaned_doc
            
            # Stage 3: Quality assessment
            quality_score = await self._assess_quality(cleaned_doc)
            cleaned_doc["quality_score"] = quality_score
            
            # Stage 4: Metadata enrichment
            cleaned_doc = await self._enrich_metadata(cleaned_doc)
            
            return cleaned_doc
            
        except Exception as e:
            logger.error(f"Error in document cleaning pipeline: {str(e)}")
            document["cleaning_error"] = str(e)
            return document
    
    async def _is_duplicate(self, document: Dict[str, Any]) -> bool:
        """
        Check if document is a duplicate
        """
        content = document.get("content", "")
        
        # Create a simple hash for duplicate detection
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self._seen_documents:
            return True
        
        self._seen_documents.add(content_hash)
        
        # Keep only recent documents in memory
        if len(self._seen_documents) > 10000:
            self._seen_documents = set(list(self._seen_documents)[-5000:])
        
        return False
    
    async def _assess_quality(self, document: Dict[str, Any]) -> float:
        """
        Assess document quality score (0.0 to 1.0)
        """
        content = document.get("content", "")
        
        if not content:
            return 0.0
        
        score = 0.0
        
        # Length score (0.3 weight)
        length_score = min(len(content) / 1000, 1.0)  # Normalize to 0-1
        score += length_score * 0.3
        
        # Vocabulary diversity (0.3 weight)
        words = content.split()
        if words:
            diversity = len(set(words)) / len(words)
            score += diversity * 0.3
        
        # Sentence structure (0.2 weight)
        sentences = content.split('.')
        if len(sentences) > 1:
            avg_sentence_length = len(content) / len(sentences)
            structure_score = min(avg_sentence_length / 20, 1.0)  # Ideal ~20 chars per sentence
            score += structure_score * 0.2
        
        # Content indicators (0.2 weight)
        indicators = ["introduction", "conclusion", "summary", "therefore", "however", "because"]
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in content.lower())
        indicator_score = min(indicator_count / len(indicators), 1.0)
        score += indicator_score * 0.2
        
        return round(score, 3)
    
    async def _enrich_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich document metadata
        """
        content = document.get("content", "")
        
        # Add basic statistics
        metadata = document.get("metadata", {})
        metadata.update({
            "char_count": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(content.split('.')),
            "cleaning_timestamp": str(datetime.utcnow()),
            "cleaner_version": "1.0.0"
        })
        
        document["metadata"] = metadata
        return document
