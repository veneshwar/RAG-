"""
Kafka producer for real-time data publishing
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError


logger = logging.getLogger(__name__)


class KafkaProducer:
    """Async Kafka producer for real-time data publishing"""
    
    def __init__(
        self,
        bootstrap_servers: str,
        client_id: Optional[str] = None,
        acks: str = "all",
        retries: int = 3,
        batch_size: int = 16384,
        linger_ms: int = 10,
        compression_type: str = "gzip",
        max_request_size: int = 1048576,  # 1MB
        enable_idempotence: bool = True
    ):
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.acks = acks
        self.retries = retries
        self.batch_size = batch_size
        self.linger_ms = linger_ms
        self.compression_type = compression_type
        self.max_request_size = max_request_size
        self.enable_idempotence = enable_idempotence
        
        self.producer = None
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                client_id=self.client_id,
                acks=self.acks,
                retries=self.retries,
                batch_size=self.batch_size,
                linger_ms=self.linger_ms,
                compression_type=self.compression_type,
                max_request_size=self.max_request_size,
                enable_idempotence=self.enable_idempotence,
                value_serializer=lambda v: json.dumps(v).encode('utf-8') if v else None,
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            await self.producer.start()
            self.is_running = True
            
            logger.info("Kafka producer started")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the Kafka producer"""
        self.is_running = False
        self._shutdown_event.set()
        
        if self.producer:
            await self.producer.stop()
        
        logger.info("Kafka producer stopped")
    
    async def send_message(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
        partition: Optional[int] = None,
        timestamp: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Send a single message to Kafka
        """
        if not self.is_running:
            logger.error("Producer not running")
            return None
        
        try:
            # Prepare headers
            kafka_headers = []
            if headers:
                for k, v in headers.items():
                    kafka_headers.append((k, v.encode('utf-8')))
            
            # Send message
            record_metadata = await self.producer.send_and_wait(
                topic=topic,
                value=message,
                key=key,
                partition=partition,
                timestamp=timestamp,
                headers=kafka_headers
            )
            
            result = {
                "topic": record_metadata.topic,
                "partition": record_metadata.partition,
                "offset": record_metadata.offset,
                "timestamp": record_metadata.timestamp
            }
            
            logger.debug(f"Message sent to {topic}: {result}")
            return result
            
        except KafkaError as e:
            logger.error(f"Kafka error sending message to {topic}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error sending message to {topic}: {str(e)}")
            return None
    
    async def send_batch(
        self,
        messages: List[Dict[str, Any]],
        default_topic: Optional[str] = None
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Send multiple messages in batch
        """
        if not self.is_running:
            logger.error("Producer not running")
            return [None] * len(messages)
        
        results = []
        
        for msg in messages:
            topic = msg.get("topic", default_topic)
            if not topic:
                logger.error("No topic specified for message")
                results.append(None)
                continue
            
            result = await self.send_message(
                topic=topic,
                message=msg.get("message", {}),
                key=msg.get("key"),
                partition=msg.get("partition"),
                timestamp=msg.get("timestamp"),
                headers=msg.get("headers")
            )
            results.append(result)
        
        return results
    
    async def send_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        topic: str = "events",
        event_id: Optional[str] = None,
        source: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Send an event message
        """
        import uuid
        import time
        
        event = {
            "event_id": event_id or str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": int(time.time() * 1000),
            "source": source or "rag-system",
            "data": event_data
        }
        
        return await self.send_message(
            topic=topic,
            message=event,
            key=event_id
        )
    
    async def send_document_event(
        self,
        document_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        topic: str = "document-events"
    ) -> Optional[Dict[str, Any]]:
        """
        Send a document-related event
        """
        return await self.send_event(
            event_type=f"document.{event_type}",
            event_data={
                "document_id": document_id,
                **event_data
            },
            topic=topic,
            event_id=document_id
        )
    
    async def send_query_event(
        self,
        query_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        topic: str = "query-events"
    ) -> Optional[Dict[str, Any]]:
        """
        Send a query-related event
        """
        return await self.send_event(
            event_type=f"query.{event_type}",
            event_data={
                "query_id": query_id,
                **event_data
            },
            topic=topic,
            event_id=query_id
        )
    
    async def send_metrics(
        self,
        metrics_data: Dict[str, Any],
        topic: str = "metrics"
    ) -> Optional[Dict[str, Any]]:
        """
        Send metrics data
        """
        import time
        
        message = {
            "timestamp": int(time.time() * 1000),
            "metrics": metrics_data
        }
        
        return await self.send_message(
            topic=topic,
            message=message
        )
    
    async def flush(self):
        """Flush any pending messages"""
        if self.producer:
            await self.producer.flush()
    
    async def get_producer_info(self) -> Dict[str, Any]:
        """Get producer information"""
        if not self.producer:
            return {"status": "not_started"}
        
        try:
            return {
                "status": "running",
                "bootstrap_servers": self.bootstrap_servers,
                "client_id": self.client_id,
                "acks": self.acks,
                "compression_type": self.compression_type,
                "enable_idempotence": self.enable_idempotence
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the producer"""
        if not self.is_running:
            return {"status": "unhealthy", "reason": "producer not running"}
        
        try:
            # Try to send a test message
            test_result = await self.send_message(
                topic="health-check",
                message={"test": True, "timestamp": asyncio.get_event_loop().time()},
                key="health-check"
            )
            
            if test_result:
                return {
                    "status": "healthy",
                    "test_message": test_result
                }
            else:
                return {"status": "unhealthy", "reason": "failed to send test message"}
                
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class DocumentKafkaProducer(KafkaProducer):
    """Specialized Kafka producer for document events"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def publish_document_ingested(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Publish document ingested event"""
        return await self.send_document_event(
            document_id=document_id,
            event_type="ingested",
            event_data=metadata
        )
    
    async def publish_document_processed(
        self,
        document_id: str,
        processing_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Publish document processed event"""
        return await self.send_document_event(
            document_id=document_id,
            event_type="processed",
            event_data=processing_result
        )
    
    async def publish_document_indexed(
        self,
        document_id: str,
        indexing_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Publish document indexed event"""
        return await self.send_document_event(
            document_id=document_id,
            event_type="indexed",
            event_data=indexing_result
        )
    
    async def publish_document_error(
        self,
        document_id: str,
        error: str,
        error_context: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Publish document error event"""
        return await self.send_document_event(
            document_id=document_id,
            event_type="error",
            event_data={
                "error": error,
                "context": error_context or {}
            }
        )
