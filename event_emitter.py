"""
Event emitter for real-time event publishing
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable, AsyncGenerator
from datetime import datetime
import uuid
from enum import Enum

from app.streaming.producers.kafka_producer import KafkaProducer


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Event types"""
    DOCUMENT_INGESTED = "document.ingested"
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_INDEXED = "document.indexed"
    DOCUMENT_ERROR = "document.error"
    QUERY_STARTED = "query.started"
    QUERY_COMPLETED = "query.completed"
    QUERY_ERROR = "query.error"
    SYSTEM_METRICS = "system.metrics"
    HEALTH_CHECK = "health.check"
    STREAM_DATA = "stream.data"


class EventEmitter:
    """Event emitter for real-time event publishing"""
    
    def __init__(
        self,
        kafka_producer: Optional[KafkaProducer] = None,
        enable_local_listeners: bool = True
    ):
        self.kafka_producer = kafka_producer
        self.enable_local_listeners = enable_local_listeners
        
        self.local_listeners: Dict[str, List[Callable]] = {}
        self.global_listeners: List[Callable] = []
        self.event_buffer: List[Dict[str, Any]] = []
        self.buffer_size = 1000
        self.is_running = False
    
    async def start(self):
        """Start the event emitter"""
        self.is_running = True
        logger.info("Event emitter started")
    
    async def stop(self):
        """Stop the event emitter"""
        self.is_running = False
        
        # Flush any pending events
        if self.kafka_producer:
            await self.kafka_producer.flush()
        
        logger.info("Event emitter stopped")
    
    async def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        event_id: Optional[str] = None,
        source: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Emit an event
        """
        if not self.is_running:
            logger.warning("Event emitter not running, event not emitted")
            return ""
        
        # Create event
        event = {
            "event_id": event_id or str(uuid.uuid4()),
            "event_type": event_type.value,
            "timestamp": timestamp or datetime.utcnow(),
            "source": source or "rag-system",
            "data": data,
            "metadata": metadata or {}
        }
        
        # Add to buffer
        self._add_to_buffer(event)
        
        # Send to Kafka if available
        if self.kafka_producer:
            try:
                await self.kafka_producer.send_event(
                    event_type=event_type.value,
                    event_data=data,
                    event_id=event["event_id"],
                    source=source
                )
            except Exception as e:
                logger.error(f"Error sending event to Kafka: {str(e)}")
        
        # Notify local listeners
        if self.enable_local_listeners:
            await self._notify_listeners(event)
        
        logger.debug(f"Emitted event: {event_type.value} - {event['event_id']}")
        return event["event_id"]
    
    async def emit_document_event(
        self,
        document_id: str,
        event_type: str,
        data: Dict[str, Any],
        **kwargs
    ) -> str:
        """Emit a document-related event"""
        full_event_type = f"document.{event_type}"
        
        return await self.emit(
            event_type=EventType(full_event_type),
            data={
                "document_id": document_id,
                **data
            },
            **kwargs
        )
    
    async def emit_query_event(
        self,
        query_id: str,
        event_type: str,
        data: Dict[str, Any],
        **kwargs
    ) -> str:
        """Emit a query-related event"""
        full_event_type = f"query.{event_type}"
        
        return await self.emit(
            event_type=EventType(full_event_type),
            data={
                "query_id": query_id,
                **data
            },
            **kwargs
        )
    
    async def emit_system_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        **kwargs
    ) -> str:
        """Emit a system-related event"""
        full_event_type = f"system.{event_type}"
        
        return await self.emit(
            event_type=EventType(full_event_type),
            data=data,
            **kwargs
        )
    
    def add_listener(self, event_type: str, listener: Callable):
        """Add a listener for specific event type"""
        if event_type not in self.local_listeners:
            self.local_listeners[event_type] = []
        
        self.local_listeners[event_type].append(listener)
        logger.info(f"Added listener for event type: {event_type}")
    
    def remove_listener(self, event_type: str, listener: Callable):
        """Remove a listener for specific event type"""
        if event_type in self.local_listeners:
            try:
                self.local_listeners[event_type].remove(listener)
                logger.info(f"Removed listener for event type: {event_type}")
            except ValueError:
                logger.warning(f"Listener not found for event type: {event_type}")
    
    def add_global_listener(self, listener: Callable):
        """Add a global listener for all events"""
        self.global_listeners.append(listener)
        logger.info("Added global event listener")
    
    def remove_global_listener(self, listener: Callable):
        """Remove a global listener"""
        try:
            self.global_listeners.remove(listener)
            logger.info("Removed global event listener")
        except ValueError:
            logger.warning("Global listener not found")
    
    async def _notify_listeners(self, event: Dict[str, Any]):
        """Notify all relevant listeners"""
        event_type = event["event_type"]
        
        # Notify specific listeners
        if event_type in self.local_listeners:
            for listener in self.local_listeners[event_type]:
                try:
                    if asyncio.iscoroutinefunction(listener):
                        await listener(event)
                    else:
                        listener(event)
                except Exception as e:
                    logger.error(f"Error in event listener: {str(e)}")
        
        # Notify global listeners
        for listener in self.global_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                logger.error(f"Error in global event listener: {str(e)}")
    
    def _add_to_buffer(self, event: Dict[str, Any]):
        """Add event to buffer"""
        self.event_buffer.append(event)
        
        # Maintain buffer size
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer = self.event_buffer[-self.buffer_size:]
    
    def get_recent_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent events from buffer"""
        events = self.event_buffer
        
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        
        return events[-limit:] if events else []
    
    async def process_stream(
        self,
        source: str,
        stream_config: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a stream of events (placeholder implementation)
        """
        # This would integrate with actual stream sources
        # For now, yield sample events
        
        for i in range(10):
            event = {
                "event_id": str(uuid.uuid4()),
                "event_type": "stream.data",
                "timestamp": datetime.utcnow(),
                "source": source,
                "data": {
                    "stream_id": stream_config.get("stream_id", "default"),
                    "sequence": i,
                    "content": f"Stream data {i}"
                }
            }
            
            await self.emit(
                event_type=EventType.STREAM_DATA,
                data=event["data"],
                source=source,
                event_id=event["event_id"]
            )
            
            yield event
            
            await asyncio.sleep(1)  # Simulate stream delay
    
    async def get_event_stats(self) -> Dict[str, Any]:
        """Get event statistics"""
        event_counts = {}
        
        for event in self.event_buffer:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "total_events": len(self.event_buffer),
            "event_types": event_counts,
            "listeners": {
                "specific": {k: len(v) for k, v in self.local_listeners.items()},
                "global": len(self.global_listeners)
            },
            "buffer_size": len(self.event_buffer),
            "max_buffer_size": self.buffer_size
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the event emitter"""
        status = {
            "status": "healthy" if self.is_running else "unhealthy",
            "is_running": self.is_running,
            "kafka_producer": None,
            "listeners": len(self.global_listeners) + sum(len(v) for v in self.local_listeners.values()),
            "buffer_size": len(self.event_buffer)
        }
        
        if self.kafka_producer:
            try:
                kafka_health = await self.kafka_producer.health_check()
                status["kafka_producer"] = kafka_health["status"]
            except Exception as e:
                status["kafka_producer"] = f"error: {str(e)}"
        
        return status


class DocumentEventEmitter(EventEmitter):
    """Specialized event emitter for document events"""
    
    async def document_ingested(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Emit document ingested event"""
        return await self.emit_document_event(
            document_id=document_id,
            event_type="ingested",
            data=metadata
        )
    
    async def document_processed(
        self,
        document_id: str,
        processing_result: Dict[str, Any]
    ) -> str:
        """Emit document processed event"""
        return await self.emit_document_event(
            document_id=document_id,
            event_type="processed",
            data=processing_result
        )
    
    async def document_indexed(
        self,
        document_id: str,
        indexing_result: Dict[str, Any]
    ) -> str:
        """Emit document indexed event"""
        return await self.emit_document_event(
            document_id=document_id,
            event_type="indexed",
            data=indexing_result
        )
    
    async def document_error(
        self,
        document_id: str,
        error: str,
        error_context: Dict[str, Any] = None
    ) -> str:
        """Emit document error event"""
        return await self.emit_document_event(
            document_id=document_id,
            event_type="error",
            data={
                "error": error,
                "context": error_context or {}
            }
        )


class QueryEventEmitter(EventEmitter):
    """Specialized event emitter for query events"""
    
    async def query_started(
        self,
        query_id: str,
        query: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Emit query started event"""
        return await self.emit_query_event(
            query_id=query_id,
            event_type="started",
            data={
                "query": query,
                "metadata": metadata or {}
            }
        )
    
    async def query_completed(
        self,
        query_id: str,
        result: Dict[str, Any]
    ) -> str:
        """Emit query completed event"""
        return await self.emit_query_event(
            query_id=query_id,
            event_type="completed",
            data=result
        )
    
    async def query_error(
        self,
        query_id: str,
        error: str,
        error_context: Dict[str, Any] = None
    ) -> str:
        """Emit query error event"""
        return await self.emit_query_event(
            query_id=query_id,
            event_type="error",
            data={
                "error": error,
                "context": error_context or {}
            }
        )
