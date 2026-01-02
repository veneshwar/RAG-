"""
Kafka consumer for real-time data ingestion
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

logger = logging.getLogger(__name__)


class KafkaConsumer:
    """Async Kafka consumer for real-time data processing"""
    
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: List[str],
        auto_offset_reset: str = "latest",
        enable_auto_commit: bool = True,
        session_timeout_ms: int = 10000,
        heartbeat_interval_ms: int = 3000,
        max_poll_records: int = 500
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics
        self.auto_offset_reset = auto_offset_reset
        self.enable_auto_commit = enable_auto_commit
        self.session_timeout_ms = session_timeout_ms
        self.heartbeat_interval_ms = heartbeat_interval_ms
        self.max_poll_records = max_poll_records
        
        self.consumer = None
        self.message_handlers = {}
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the Kafka consumer"""
        try:
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=self.enable_auto_commit,
                session_timeout_ms=self.session_timeout_ms,
                heartbeat_interval_ms=self.heartbeat_interval_ms,
                max_poll_records=self.max_poll_records,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')) if v else None
            )
            
            await self.consumer.start()
            self.is_running = True
            
            logger.info(f"Kafka consumer started for topics: {self.topics}")
            
            # Start consuming messages
            asyncio.create_task(self._consume_messages())
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the Kafka consumer"""
        self.is_running = False
        self._shutdown_event.set()
        
        if self.consumer:
            await self.consumer.stop()
        
        logger.info("Kafka consumer stopped")
    
    async def _consume_messages(self):
        """Main message consumption loop"""
        try:
            while self.is_running and not self._shutdown_event.is_set():
                try:
                    # Poll for messages with timeout
                    msg_pack = await asyncio.wait_for(
                        self.consumer.getmany(timeout_ms=1000),
                        timeout=1.0
                    )
                    
                    for topic_partition, messages in msg_pack.items():
                        for message in messages:
                            await self._process_message(message)
                
                except asyncio.TimeoutError:
                    # No messages available, continue
                    continue
                except KafkaError as e:
                    logger.error(f"Kafka error: {str(e)}")
                    await asyncio.sleep(1)  # Backoff on error
                except Exception as e:
                    logger.error(f"Unexpected error in consumer loop: {str(e)}")
                    await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Fatal error in consumer loop: {str(e)}")
            self.is_running = False
    
    async def _process_message(self, message):
        """Process a single message"""
        try:
            topic = message.topic
            value = message.value
            
            if not value:
                logger.warning(f"Received empty message from topic {topic}")
                return
            
            # Get handler for this topic
            handler = self.message_handlers.get(topic)
            if handler:
                try:
                    await handler(value)
                except Exception as e:
                    logger.error(f"Error processing message from topic {topic}: {str(e)}")
            else:
                logger.warning(f"No handler registered for topic {topic}")
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def register_handler(self, topic: str, handler: Callable):
        """Register a message handler for a specific topic"""
        self.message_handlers[topic] = handler
        logger.info(f"Registered handler for topic: {topic}")
    
    def unregister_handler(self, topic: str):
        """Unregister a message handler"""
        if topic in self.message_handlers:
            del self.message_handlers[topic]
            logger.info(f"Unregistered handler for topic: {topic}")
    
    async def commit_offset(self, topic: str, partition: int, offset: int):
        """Manually commit offset"""
        try:
            from aiokafka.structs import TopicPartition
            tp = TopicPartition(topic, partition)
            await self.consumer.commit({tp: offset})
        except Exception as e:
            logger.error(f"Error committing offset: {str(e)}")
    
    async def get_consumer_info(self) -> Dict[str, Any]:
        """Get consumer information"""
        if not self.consumer:
            return {"status": "not_started"}
        
        try:
            assignment = self.consumer.assignment()
            return {
                "status": "running",
                "topics": self.topics,
                "group_id": self.group_id,
                "assignment": [(tp.topic, tp.partition) for tp in assignment],
                "handlers": list(self.message_handlers.keys())
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the consumer"""
        if not self.is_running:
            return {"status": "unhealthy", "reason": "consumer not running"}
        
        try:
            # Check if consumer is responsive
            info = await self.get_consumer_info()
            return {
                "status": "healthy",
                "info": info
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class DocumentIngestionConsumer(KafkaConsumer):
    """Specialized consumer for document ingestion"""
    
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        document_processor: Callable,
        **kwargs
    ):
        super().__init__(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            topics=["document-ingestion"],
            **kwargs
        )
        self.document_processor = document_processor
        
        # Register default handler
        self.register_handler("document-ingestion", self._handle_document_ingestion)
    
    async def _handle_document_ingestion(self, message: Dict[str, Any]):
        """Handle document ingestion messages"""
        try:
            document_id = message.get("document_id")
            content = message.get("content")
            metadata = message.get("metadata", {})
            source = message.get("source", "unknown")
            
            if not document_id or not content:
                logger.error(f"Invalid document message: {message}")
                return
            
            logger.info(f"Processing document {document_id} from {source}")
            
            # Process the document
            result = await self.document_processor(
                document_id=document_id,
                content=content,
                metadata=metadata,
                source=source
            )
            
            logger.info(f"Processed document {document_id}: {result}")
            
        except Exception as e:
            logger.error(f"Error processing document {message.get('document_id')}: {str(e)}")
            raise


class StreamDataConsumer(KafkaConsumer):
    """Specialized consumer for streaming data"""
    
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        stream_processor: Callable,
        topics: List[str],
        **kwargs
    ):
        super().__init__(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            topics=topics,
            **kwargs
        )
        self.stream_processor = stream_processor
        
        # Register handlers for all topics
        for topic in topics:
            self.register_handler(topic, self._handle_stream_data)
    
    async def _handle_stream_data(self, message: Dict[str, Any]):
        """Handle streaming data messages"""
        try:
            stream_id = message.get("stream_id")
            data = message.get("data")
            timestamp = message.get("timestamp")
            
            if not stream_id or not data:
                logger.error(f"Invalid stream message: {message}")
                return
            
            # Process the stream data
            result = await self.stream_processor(
                stream_id=stream_id,
                data=data,
                timestamp=timestamp
            )
            
            logger.debug(f"Processed stream data {stream_id}: {result}")
            
        except Exception as e:
            logger.error(f"Error processing stream data {message.get('stream_id')}: {str(e)}")
            raise
