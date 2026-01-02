"""
RSS consumer for real-time data ingestion
"""

import asyncio
import logging
from typing import Dict, Any, Callable, Optional, List
import feedparser
from datetime import datetime, timezone
import aiohttp
from urllib.parse import urljoin, urlparse


logger = logging.getLogger(__name__)


class RSSConsumer:
    """RSS consumer for real-time data processing"""
    
    def __init__(
        self,
        feeds: List[str],
        poll_interval: int = 300,  # 5 minutes
        user_agent: str = "Real-time-RAG-RSS-Consumer/1.0",
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 60
    ):
        self.feeds = feeds
        self.poll_interval = poll_interval
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.session = None
        self.message_handlers = {}
        self.processed_entries = set()  # Track processed entries to avoid duplicates
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the RSS consumer"""
        try:
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            self.is_running = True
            
            logger.info(f"RSS consumer started for {len(self.feeds)} feeds")
            
            # Start polling feeds
            asyncio.create_task(self._poll_feeds())
            
        except Exception as e:
            logger.error(f"Failed to start RSS consumer: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the RSS consumer"""
        self.is_running = False
        self._shutdown_event.set()
        
        if self.session:
            await self.session.close()
        
        logger.info("RSS consumer stopped")
    
    async def _poll_feeds(self):
        """Main feed polling loop"""
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Poll all feeds concurrently
                tasks = [self._poll_feed(feed) for feed in self.feeds]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait for next poll
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.poll_interval
                )
                
            except asyncio.TimeoutError:
                # Normal timeout, continue polling
                continue
            except Exception as e:
                logger.error(f"Error in feed polling loop: {str(e)}")
                await asyncio.sleep(60)  # Backoff on error
    
    async def _poll_feed(self, feed_url: str):
        """Poll a single RSS feed"""
        for attempt in range(self.max_retries):
            try:
                async with self.session.get(feed_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        await self._process_feed_content(feed_url, content)
                        return
                    else:
                        logger.warning(f"HTTP {response.status} for feed {feed_url}")
                        
            except Exception as e:
                logger.error(f"Error polling feed {feed_url} (attempt {attempt + 1}): {str(e)}")
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to poll feed {feed_url} after {self.max_retries} attempts")
    
    async def _process_feed_content(self, feed_url: str, content: str):
        """Process RSS feed content"""
        try:
            feed = feedparser.parse(content)
            
            if feed.bozo:
                logger.warning(f"Feed parsing warning for {feed_url}: {feed.bozo_exception}")
            
            # Process entries
            for entry in feed.entries:
                await self._process_entry(feed_url, entry)
                
        except Exception as e:
            logger.error(f"Error processing feed content from {feed_url}: {str(e)}")
    
    async def _process_entry(self, feed_url: str, entry):
        """Process a single RSS entry"""
        try:
            # Create unique identifier for entry
            entry_id = entry.get("id") or entry.get("link") or str(hash(entry.get("title", "")))
            
            # Skip if already processed
            if entry_id in self.processed_entries:
                return
            
            # Extract entry data
            entry_data = {
                "feed_url": feed_url,
                "entry_id": entry_id,
                "title": entry.get("title", ""),
                "link": entry.get("link", ""),
                "description": entry.get("description", ""),
                "content": self._extract_content(entry),
                "author": entry.get("author", ""),
                "published": self._parse_date(entry.get("published")),
                "updated": self._parse_date(entry.get("updated")),
                "tags": self._extract_tags(entry),
                "metadata": {
                    "source": "rss",
                    "feed_url": feed_url,
                    "original_entry": dict(entry)
                }
            }
            
            # Mark as processed
            self.processed_entries.add(entry_id)
            
            # Get handler for RSS entries
            handler = self.message_handlers.get("rss_entry")
            if handler:
                try:
                    await handler(entry_data)
                except Exception as e:
                    logger.error(f"Error processing RSS entry {entry_id}: {str(e)}")
            else:
                logger.warning(f"No handler registered for RSS entries")
            
            # Clean up old entries (keep last 1000)
            if len(self.processed_entries) > 1000:
                self.processed_entries = set(list(self.processed_entries)[-500:])
            
        except Exception as e:
            logger.error(f"Error processing RSS entry: {str(e)}")
    
    def _extract_content(self, entry) -> str:
        """Extract content from RSS entry"""
        # Try different content fields
        content_fields = ["content", "summary", "description"]
        
        for field in content_fields:
            if hasattr(entry, field):
                value = getattr(entry, field)
                if isinstance(value, list) and value:
                    return value[0].value if hasattr(value[0], "value") else str(value[0])
                elif isinstance(value, str):
                    return value
        
        return ""
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        try:
            # Try to parse with feedparser
            parsed = feedparser._parse_date(date_str)
            if parsed:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
        except:
            pass
        
        try:
            # Fallback to datetime parsing
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except:
            pass
        
        return None
    
    def _extract_tags(self, entry) -> List[str]:
        """Extract tags from RSS entry"""
        tags = []
        
        if hasattr(entry, "tags") and entry.tags:
            for tag in entry.tags:
                if hasattr(tag, "term"):
                    tags.append(tag.term)
                elif isinstance(tag, str):
                    tags.append(tag)
        
        return tags
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    def unregister_handler(self, message_type: str):
        """Unregister a message handler"""
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
            logger.info(f"Unregistered handler for message type: {message_type}")
    
    def add_feed(self, feed_url: str):
        """Add a new RSS feed"""
        if feed_url not in self.feeds:
            self.feeds.append(feed_url)
            logger.info(f"Added RSS feed: {feed_url}")
    
    def remove_feed(self, feed_url: str):
        """Remove an RSS feed"""
        if feed_url in self.feeds:
            self.feeds.remove(feed_url)
            logger.info(f"Removed RSS feed: {feed_url}")
    
    async def get_consumer_info(self) -> Dict[str, Any]:
        """Get consumer information"""
        return {
            "status": "running" if self.is_running else "stopped",
            "feeds": self.feeds,
            "poll_interval": self.poll_interval,
            "processed_entries": len(self.processed_entries),
            "handlers": list(self.message_handlers.keys())
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the RSS consumer"""
        if not self.is_running:
            return {"status": "unhealthy", "reason": "consumer not running"}
        
        try:
            # Test connectivity to feeds
            healthy_feeds = 0
            for feed_url in self.feeds[:3]:  # Test first 3 feeds
                try:
                    async with self.session.head(feed_url) as response:
                        if response.status < 400:
                            healthy_feeds += 1
                except:
                    pass
            
            status = "healthy" if healthy_feeds > 0 else "degraded"
            
            return {
                "status": status,
                "healthy_feeds": healthy_feeds,
                "total_feeds": len(self.feeds),
                "info": await self.get_consumer_info()
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class DocumentRSSConsumer(RSSConsumer):
    """Specialized RSS consumer for document ingestion"""
    
    def __init__(self, document_processor: Callable, **kwargs):
        super().__init__(**kwargs)
        self.document_processor = document_processor
        
        # Register default handler
        self.register_handler("rss_entry", self._handle_rss_entry)
    
    async def _handle_rss_entry(self, entry_data: Dict[str, Any]):
        """Handle RSS entry as document"""
        try:
            document_id = f"rss_{entry_data['entry_id']}"
            content = entry_data["content"] or entry_data["description"]
            
            if not content:
                logger.warning(f"No content for RSS entry {entry_data['entry_id']}")
                return
            
            # Create metadata
            metadata = {
                "title": entry_data["title"],
                "link": entry_data["link"],
                "author": entry_data["author"],
                "published": entry_data["published"].isoformat() if entry_data["published"] else None,
                "tags": entry_data["tags"],
                "feed_url": entry_data["feed_url"],
                **entry_data["metadata"]
            }
            
            logger.info(f"Processing RSS document {document_id}")
            
            # Process the document
            result = await self.document_processor(
                document_id=document_id,
                content=content,
                metadata=metadata,
                source="rss"
            )
            
            logger.info(f"Processed RSS document {document_id}: {result}")
            
        except Exception as e:
            logger.error(f"Error processing RSS entry {entry_data.get('entry_id')}: {str(e)}")
            raise
