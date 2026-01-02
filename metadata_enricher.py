"""
Metadata enricher for streaming document processing
"""

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import hashlib
import json


logger = logging.getLogger(__name__)


class MetadataEnricher:
    """Metadata enricher for adding additional information to documents"""
    
    def __init__(
        self,
        extract_entities: bool = True,
        extract_keywords: bool = True,
        calculate_readability: bool = True,
        detect_language: bool = True,
        add_statistics: bool = True,
        add_hashes: bool = True,
        custom_extractors: Optional[List[callable]] = None
    ):
        self.extract_entities = extract_entities
        self.extract_keywords = extract_keywords
        self.calculate_readability = calculate_readability
        self.detect_language = detect_language
        self.add_statistics = add_statistics
        self.add_hashes = add_hashes
        self.custom_extractors = custom_extractors or []
        
        # Compile regex patterns
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        self.date_pattern = re.compile(r'\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b')
        self.number_pattern = re.compile(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b')
    
    async def enrich_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich document metadata with additional information
        """
        try:
            content = document.get("content", "")
            if not content:
                return document
            
            # Get existing metadata
            metadata = document.get("metadata", {}).copy()
            
            # Add enrichment timestamp
            metadata["enrichment_timestamp"] = datetime.now(timezone.utc).isoformat()
            metadata["enricher_version"] = "1.0.0"
            
            # Extract entities
            if self.extract_entities:
                entities = await self._extract_entities(content)
                metadata["entities"] = entities
            
            # Extract keywords
            if self.extract_keywords:
                keywords = await self._extract_keywords(content)
                metadata["keywords"] = keywords
            
            # Calculate readability
            if self.calculate_readability:
                readability = await self._calculate_readability(content)
                metadata["readability"] = readability
            
            # Detect language
            if self.detect_language:
                language = await self._detect_language(content)
                metadata["language"] = language
            
            # Add statistics
            if self.add_statistics:
                statistics = await self._calculate_statistics(content)
                metadata["statistics"] = statistics
            
            # Add hashes
            if self.add_hashes:
                hashes = await self._calculate_hashes(content)
                metadata["hashes"] = hashes
            
            # Apply custom extractors
            for extractor in self.custom_extractors:
                try:
                    custom_data = await extractor(content, metadata)
                    metadata.update(custom_data)
                except Exception as e:
                    logger.error(f"Error in custom extractor: {str(e)}")
            
            # Update document
            enriched_document = document.copy()
            enriched_document["metadata"] = metadata
            
            return enriched_document
            
        except Exception as e:
            logger.error(f"Error enriching metadata: {str(e)}")
            document["enrichment_error"] = str(e)
            return document
    
    async def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """
        Extract entities from content
        """
        entities = {
            "emails": [],
            "urls": [],
            "phone_numbers": [],
            "dates": [],
            "numbers": [],
            "mentions": [],  # @mentions
            "hashtags": []  # #hashtags
        }
        
        try:
            # Extract emails
            entities["emails"] = list(set(self.email_pattern.findall(content)))
            
            # Extract URLs
            entities["urls"] = list(set(self.url_pattern.findall(content)))
            
            # Extract phone numbers
            entities["phone_numbers"] = list(set(self.phone_pattern.findall(content)))
            
            # Extract dates
            entities["dates"] = list(set(self.date_pattern.findall(content)))
            
            # Extract numbers
            entities["numbers"] = list(set(self.number_pattern.findall(content)))
            
            # Extract mentions (@username)
            mention_pattern = re.compile(r'@\w+')
            entities["mentions"] = list(set(mention_pattern.findall(content)))
            
            # Extract hashtags
            hashtag_pattern = re.compile(r'#\w+')
            entities["hashtags"] = list(set(hashtag_pattern.findall(content)))
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
        
        return entities
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """
        Extract keywords from content (simplified implementation)
        """
        try:
            # Simple keyword extraction based on word frequency
            words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
            
            # Filter out common stop words
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
                'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
                'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
                'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 'just'
            }
            
            # Filter stop words and count frequency
            filtered_words = [word for word in words if word not in stop_words]
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top keywords
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            keywords = [word for word, freq in sorted_words[:10]]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    async def _calculate_readability(self, content: str) -> Dict[str, float]:
        """
        Calculate readability metrics
        """
        try:
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            words = content.split()
            syllables = sum(self._count_syllables(word) for word in words)
            
            if not sentences or not words:
                return {"flesch_score": 0.0, "flesch_grade": 0.0, "avg_sentence_length": 0.0}
            
            # Flesch Reading Ease Score
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = syllables / len(words)
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Flesch-Kincaid Grade Level
            flesch_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            
            return {
                "flesch_score": round(flesch_score, 2),
                "flesch_grade": round(flesch_grade, 2),
                "avg_sentence_length": round(avg_sentence_length, 2),
                "avg_syllables_per_word": round(avg_syllables_per_word, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating readability: {str(e)}")
            return {"flesch_score": 0.0, "flesch_grade": 0.0, "avg_sentence_length": 0.0}
    
    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word (simplified implementation)
        """
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_char_was_vowel:
                syllable_count += 1
            prev_char_was_vowel = is_vowel
        
        # Handle silent 'e' at the end
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(syllable_count, 1)
    
    async def _detect_language(self, content: str) -> Dict[str, Any]:
        """
        Detect document language (simplified implementation)
        """
        try:
            # Simple language detection based on common words
            # In production, use libraries like langdetect or polyglot
            
            content_lower = content.lower()
            
            # Common words in different languages
            language_indicators = {
                "en": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"],
                "es": ["el", "la", "de", "que", "y", "a", "en", "un", "es", "se"],
                "fr": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
                "de": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"],
                "it": ["il", "di", "che", "e", "la", "un", "a", "per", "non", "in"]
            }
            
            language_scores = {}
            for lang, indicators in language_indicators.items():
                score = sum(1 for word in indicators if word in content_lower)
                language_scores[lang] = score
            
            # Get language with highest score
            detected_lang = max(language_scores, key=language_scores.get)
            confidence = language_scores[detected_lang] / sum(language_scores.values()) if sum(language_scores.values()) > 0 else 0.0
            
            return {
                "language": detected_lang,
                "confidence": round(confidence, 3),
                "scores": language_scores
            }
            
        except Exception as e:
            logger.error(f"Error detecting language: {str(e)}")
            return {"language": "unknown", "confidence": 0.0, "scores": {}}
    
    async def _calculate_statistics(self, content: str) -> Dict[str, Any]:
        """
        Calculate document statistics
        """
        try:
            sentences = re.split(r'[.!?]+', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            words = content.split()
            paragraphs = content.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # Character statistics
            char_count = len(content)
            char_count_no_spaces = len(content.replace(' ', ''))
            
            # Word statistics
            word_count = len(words)
            unique_words = len(set(word.lower() for word in words))
            
            # Sentence statistics
            sentence_count = len(sentences)
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # Paragraph statistics
            paragraph_count = len(paragraphs)
            avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
            
            return {
                "char_count": char_count,
                "char_count_no_spaces": char_count_no_spaces,
                "word_count": word_count,
                "unique_words": unique_words,
                "vocabulary_diversity": round(unique_words / word_count, 3) if word_count > 0 else 0.0,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "paragraph_count": paragraph_count,
                "avg_paragraph_length": round(avg_paragraph_length, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {str(e)}")
            return {}
    
    async def _calculate_hashes(self, content: str) -> Dict[str, str]:
        """
        Calculate various hashes of the content
        """
        try:
            return {
                "md5": hashlib.md5(content.encode()).hexdigest(),
                "sha1": hashlib.sha1(content.encode()).hexdigest(),
                "sha256": hashlib.sha256(content.encode()).hexdigest()
            }
        except Exception as e:
            logger.error(f"Error calculating hashes: {str(e)}")
            return {}
    
    async def enrich_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich multiple documents in batch
        """
        enriched_docs = []
        
        for doc in documents:
            enriched_doc = await self.enrich_metadata(doc)
            enriched_docs.append(enriched_doc)
        
        return enriched_docs
    
    async def get_enrichment_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get enrichment statistics
        """
        stats = {
            "total_documents": len(documents),
            "enriched_documents": 0,
            "failed_enrichment": 0,
            "total_entities": 0,
            "total_keywords": 0,
            "avg_readability_score": 0.0,
            "language_distribution": {}
        }
        
        total_flesch = 0.0
        readability_count = 0
        
        for doc in documents:
            try:
                enriched_doc = await self.enrich_metadata(doc)
                
                if "enrichment_error" not in enriched_doc:
                    stats["enriched_documents"] += 1
                    
                    metadata = enriched_doc.get("metadata", {})
                    
                    # Count entities
                    if "entities" in metadata:
                        entity_count = sum(len(entities) for entities in metadata["entities"].values())
                        stats["total_entities"] += entity_count
                    
                    # Count keywords
                    if "keywords" in metadata:
                        stats["total_keywords"] += len(metadata["keywords"])
                    
                    # Readability scores
                    if "readability" in metadata:
                        flesch_score = metadata["readability"].get("flesch_score", 0.0)
                        total_flesch += flesch_score
                        readability_count += 1
                    
                    # Language distribution
                    if "language" in metadata:
                        lang = metadata["language"].get("language", "unknown")
                        stats["language_distribution"][lang] = stats["language_distribution"].get(lang, 0) + 1
                else:
                    stats["failed_enrichment"] += 1
                    
            except Exception as e:
                logger.error(f"Error in enrichment stats: {str(e)}")
                stats["failed_enrichment"] += 1
        
        # Calculate averages
        if readability_count > 0:
            stats["avg_readability_score"] = round(total_flesch / readability_count, 2)
        
        return stats


class StreamingMetadataEnricher(MetadataEnricher):
    """Metadata enricher optimized for streaming data"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._batch_buffer = []
        self._batch_size = 10
    
    async def enrich_stream_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich documents in streaming batches
        """
        self._batch_buffer.extend(documents)
        
        if len(self._batch_buffer) >= self._batch_size:
            enriched_batch = await self.enrich_batch(self._batch_buffer)
            self._batch_buffer = []
            return enriched_batch
        
        return []
    
    async def flush_buffer(self) -> List[Dict[str, Any]]:
        """
        Enrich remaining documents in buffer
        """
        if self._batch_buffer:
            enriched_batch = await self.enrich_batch(self._batch_buffer)
            self._batch_buffer = []
            return enriched_batch
        
        return []
    
    def reset_buffer(self):
        """Reset the internal buffer"""
        self._batch_buffer = []
