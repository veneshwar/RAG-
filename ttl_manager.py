"""
TTL (Time To Live) manager for automatic data expiration
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, Set
from datetime import datetime, timedelta
import time
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class TTLItem:
    """Represents an item with TTL"""
    key: str
    data: Any
    expires_at: float
    created_at: float
    metadata: Optional[Dict[str, Any]] = None


class TTLManager:
    """Manages TTL-based expiration of data"""
    
    def __init__(
        self,
        cleanup_interval: int = 60,  # seconds
        default_ttl: int = 3600,  # 1 hour
        max_items: int = 10000
    ):
        self.cleanup_interval = cleanup_interval
        self.default_ttl = default_ttl
        self.max_items = max_items
        
        self.items: Dict[str, TTLItem] = {}
        self.expiration_callbacks: Dict[str, List[Callable]] = {}
        self.is_running = False
        self._cleanup_task = None
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "total_items": 0,
            "expired_items": 0,
            "cleanup_runs": 0,
            "last_cleanup": None
        }
    
    async def start(self):
        """Start the TTL manager"""
        if self.is_running:
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"TTL manager started with {self.cleanup_interval}s cleanup interval")
    
    async def stop(self):
        """Stop the TTL manager"""
        self.is_running = False
        self._shutdown_event.set()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("TTL manager stopped")
    
    async def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set an item with TTL
        """
        try:
            current_time = time.time()
            ttl = ttl or self.default_ttl
            
            # Check if we need to remove old items
            if len(self.items) >= self.max_items and key not in self.items:
                await self._remove_oldest_item()
            
            item = TTLItem(
                key=key,
                data=data,
                expires_at=current_time + ttl,
                created_at=current_time,
                metadata=metadata
            )
            
            self.items[key] = item
            self.stats["total_items"] = len(self.items)
            
            logger.debug(f"Set TTL item {key} with {ttl}s TTL")
            return True
            
        except Exception as e:
            logger.error(f"Error setting TTL item: {str(e)}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get an item if it hasn't expired
        """
        try:
            if key not in self.items:
                return None
            
            item = self.items[key]
            current_time = time.time()
            
            if current_time > item.expires_at:
                # Item has expired, remove it
                await self._remove_item(key, "expired_on_access")
                return None
            
            return item.data
            
        except Exception as e:
            logger.error(f"Error getting TTL item: {str(e)}")
            return None
    
    async def get_item(self, key: str) -> Optional[TTLItem]:
        """
        Get the full TTL item
        """
        try:
            if key not in self.items:
                return None
            
            item = self.items[key]
            current_time = time.time()
            
            if current_time > item.expires_at:
                await self._remove_item(key, "expired_on_access")
                return None
            
            return item
            
        except Exception as e:
            logger.error(f"Error getting TTL item: {str(e)}")
            return None
    
    async def update_ttl(self, key: str, ttl: int) -> bool:
        """
        Update TTL for an existing item
        """
        try:
            if key not in self.items:
                return False
            
            item = self.items[key]
            current_time = time.time()
            
            # Update expiration time
            item.expires_at = current_time + ttl
            
            logger.debug(f"Updated TTL for item {key} to {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Error updating TTL: {str(e)}")
            return False
    
    async def extend_ttl(self, key: str, additional_seconds: int) -> bool:
        """
        Extend TTL for an existing item
        """
        try:
            if key not in self.items:
                return False
            
            item = self.items[key]
            item.expires_at += additional_seconds
            
            logger.debug(f"Extended TTL for item {key} by {additional_seconds}s")
            return True
            
        except Exception as e:
            logger.error(f"Error extending TTL: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete an item
        """
        try:
            if key not in self.items:
                return False
            
            await self._remove_item(key, "manual_delete")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting TTL item: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if an item exists and hasn't expired
        """
        item = await self.get_item(key)
        return item is not None
    
    async def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for an item
        """
        try:
            item = await self.get_item(key)
            if not item:
                return None
            
            current_time = time.time()
            remaining_ttl = int(item.expires_at - current_time)
            return max(0, remaining_ttl)
            
        except Exception as e:
            logger.error(f"Error getting TTL: {str(e)}")
            return None
    
    async def add_expiration_callback(self, key: str, callback: Callable):
        """
        Add a callback to be called when an item expires
        """
        if key not in self.expiration_callbacks:
            self.expiration_callbacks[key] = []
        
        self.expiration_callbacks[key].append(callback)
        logger.debug(f"Added expiration callback for item {key}")
    
    async def remove_expiration_callback(self, key: str, callback: Callable):
        """
        Remove an expiration callback
        """
        if key in self.expiration_callbacks:
            try:
                self.expiration_callbacks[key].remove(callback)
                logger.debug(f"Removed expiration callback for item {key}")
            except ValueError:
                pass
    
    async def _cleanup_loop(self):
        """
        Main cleanup loop
        """
        while self.is_running and not self._shutdown_event.is_set():
            try:
                await self._cleanup_expired_items()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(1)
    
    async def _cleanup_expired_items(self):
        """
        Remove expired items
        """
        current_time = time.time()
        expired_keys = []
        
        for key, item in self.items.items():
            if current_time > item.expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self._remove_item(key, "expired_cleanup")
        
        if expired_keys:
            self.stats["expired_items"] += len(expired_keys)
            logger.debug(f"Cleaned up {len(expired_keys)} expired items")
        
        self.stats["cleanup_runs"] += 1
        self.stats["last_cleanup"] = datetime.utcnow().isoformat()
        self.stats["total_items"] = len(self.items)
    
    async def _remove_item(self, key: str, reason: str):
        """
        Remove an item and trigger callbacks
        """
        if key not in self.items:
            return
        
        item = self.items.pop(key)
        
        # Trigger expiration callbacks
        if key in self.expiration_callbacks:
            for callback in self.expiration_callbacks[key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key, item.data, reason, item.metadata)
                    else:
                        callback(key, item.data, reason, item.metadata)
                except Exception as e:
                    logger.error(f"Error in expiration callback: {str(e)}")
            
            # Remove callbacks for this key
            del self.expiration_callbacks[key]
        
        self.stats["total_items"] = len(self.items)
        logger.debug(f"Removed item {key} ({reason})")
    
    async def _remove_oldest_item(self):
        """
        Remove the oldest item to make space
        """
        if not self.items:
            return
        
        oldest_key = min(
            self.items.keys(),
            key=lambda k: self.items[k].created_at
        )
        
        await self._remove_item(oldest_key, "lru_eviction")
        logger.info(f"Removed oldest item {oldest_key} to make space")
    
    async def get_all_keys(self) -> Set[str]:
        """
        Get all non-expired keys
        """
        current_time = time.time()
        valid_keys = set()
        
        for key, item in self.items.items():
            if current_time <= item.expires_at:
                valid_keys.add(key)
        
        return valid_keys
    
    async def get_items_by_pattern(self, pattern: str) -> Dict[str, Any]:
        """
        Get items matching a pattern
        """
        import re
        
        try:
            regex = re.compile(pattern)
            matching_items = {}
            
            for key, item in self.items.items():
                if regex.match(key):
                    current_time = time.time()
                    if current_time <= item.expires_at:
                        matching_items[key] = item.data
            
            return matching_items
            
        except Exception as e:
            logger.error(f"Error getting items by pattern: {str(e)}")
            return {}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get TTL manager statistics
        """
        current_time = time.time()
        
        # Calculate TTL distribution
        ttl_distribution = {
            "expired": 0,
            "expiring_soon": 0,  # < 5 minutes
            "normal": 0
        }
        
        for item in self.items.values():
            if current_time > item.expires_at:
                ttl_distribution["expired"] += 1
            elif item.expires_at - current_time < 300:  # 5 minutes
                ttl_distribution["expiring_soon"] += 1
            else:
                ttl_distribution["normal"] += 1
        
        return {
            **self.stats,
            "ttl_distribution": ttl_distribution,
            "is_running": self.is_running,
            "cleanup_interval": self.cleanup_interval,
            "default_ttl": self.default_ttl,
            "max_items": self.max_items
        }
    
    async def clear_all(self):
        """
        Clear all items
        """
        keys_to_remove = list(self.items.keys())
        for key in keys_to_remove:
            await self._remove_item(key, "clear_all")
        
        logger.info("Cleared all TTL items")


class DocumentTTLManager(TTLManager):
    """
    Specialized TTL manager for documents
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.document_callbacks = []
    
    async def store_document_with_ttl(
        self,
        document_id: str,
        document_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store a document with TTL
        """
        return await self.set(
            key=f"doc:{document_id}",
            data=document_data,
            ttl=ttl,
            metadata={"type": "document", "document_id": document_id}
        )
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document if it hasn't expired
        """
        return await self.get(f"doc:{document_id}")
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document
        """
        return await self.delete(f"doc:{document_id}")
    
    async def extend_document_ttl(self, document_id: str, additional_seconds: int) -> bool:
        """
        Extend TTL for a document
        """
        return await self.extend_ttl(f"doc:{document_id}", additional_seconds)
    
    def add_document_expiration_callback(self, callback: Callable):
        """
        Add a callback for document expiration
        """
        self.document_callbacks.append(callback)
    
    async def _remove_item(self, key: str, reason: str):
        """
        Remove item and trigger document-specific callbacks
        """
        # Check if this is a document
        if key.startswith("doc:"):
            document_id = key[4:]  # Remove "doc:" prefix
            
            # Trigger document callbacks
            for callback in self.document_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(document_id, reason)
                    else:
                        callback(document_id, reason)
                except Exception as e:
                    logger.error(f"Error in document expiration callback: {str(e)}")
        
        # Call parent method
        await super()._remove_item(key, reason)
