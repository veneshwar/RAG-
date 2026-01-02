"""
Pulsar consumer for real-time data ingestion
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List
import pulsar
from pulsar import Client, ConsumerType


logger = logging.getLogger(__name__)


class PulsarConsumer:
    """Async Pulsar consumer for real-time data processing"""
    
    def __init__(
        self,
        service_url: str,
        subscription_name: str,
        topics: List[str],
        subscription_type: str = "shared",
        consumer_name: Optional[str] = None,
        ack_timeout_ms: int = 30000,
        negative_ack_redelivery_delay_ms: int = 60000,
        max_pending_chunked_message: int = 100
    ):
        self.service_url = service_url
        self.subscription_name = subscription_name
        self.topics = topics
        self.subscription_type = subscription_type
        self.consumer_name = consumer_name
        self.ack_timeout_ms = ack_timeout_ms
        self.negative_ack_redelivery_delay_ms = negative_ack_redelivery_delay_ms
        self.max_pending_chunked_message = max_pending_chunked_message
        
        self.client = None
        self.consumer = None
        self.message_handlers = {}
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the Pulsar consumer"""
        try:
            # Create Pulsar client
            self.client = Client(self.service_url)
            
            # Create consumer
            self.consumer = self.client.subscribe(
                topic=self.topics,
                subscription_name=self.subscription_name,
                subscription_type=getattr(ConsumerType, self.subscription_type.upper()),
                consumer_name=self.consumer_name,
                ack_timeout_ms=self.ack_timeout_ms,
                negative_ack_redelivery_delay_ms=self.negative_ack_redelivery_delay_ms,
                max_pending_chunked_message=self.max_pending_chunked_message
            )
            
            self.is_running = True
            
            logger.info(f"Pulsar consumer started for topics: {self.topics}")
            
            # Start consuming messages
            asyncio.create_task(self._consume_messages())
            
        except Exception as e:
            logger.error(f"Failed to start Pulsar consumer: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the Pulsar consumer"""
        self.is_running = False
        self._shutdown_event.set()
        
        if self.consumer:
            self.consumer.close()
        
        if self.client:
            self.client.close()
        
        logger.info("Pulsar consumer stopped")
    
    async def _consume_messages(self):
        """Main message consumption loop"""
        try:
            while self.is_running and not self._shutdown_event.is_set():
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(
                        self._receive_message(),
                        timeout=1.0
                    )
                    
                    if message:
                        await self._process_message(message)
                
                except asyncio.TimeoutError:
                    # No messages available, continue
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error in consumer loop: {str(e)}")
                    await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Fatal error in consumer loop: {str(e)}")
            self.is_running = False
    
    async def _receive_message(self):
        """Receive a message from Pulsar"""
        loop = asyncio.get_event_loop()
        
        def _receive():
            try:
                return self.consumer.receive(timeout_millis=1000)
            except Exception:
                return None
        
        return await loop.run_in_executor(None, _receive)
    
    async def _process_message(self, message):
        """Process a single message"""
        try:
            topic = message.topic()
            data = json.loads(message.data().decode('utf-8'))
            
            if not data:
                logger.warning(f"Received empty message from topic {topic}")
                await self._ack_message(message)
                return
            
            # Get handler for this topic
            handler = self.message_handlers.get(topic)
            if handler:
                try:
                    await handler(data)
                    await self._ack_message(message)
                except Exception as e:
                    logger.error(f"Error processing message from topic {topic}: {str(e)}")
                    await self._nack_message(message)
            else:
                logger.warning(f"No handler registered for topic {topic}")
                await self._ack_message(message)
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await self._nack_message(message)
    
    async def _ack_message(self, message):
        """Acknowledge a message"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, message.acknowledge)
    
    async def _nack_message(self, message):
        """Negative acknowledge a message"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, message.negative_acknowledge)
    
    def register_handler(self, topic: str, handler: Callable):
        """Register a message handler for a specific topic"""
        self.message_handlers[topic] = handler
        logger.info(f"Registered handler for topic: {topic}")
    
    def unregister_handler(self, topic: str):
        """Unregister a message handler"""
        if topic in self.message_handlers:
            del self.message_handlers[topic]
            logger.info(f"Unregistered handler for topic: {topic}")
    
    async def get_consumer_info(self) -> Dict[str, Any]:
        """Get consumer information"""
        if not self.consumer:
            return {"status": "not_started"}
        
        try:
            return {
                "status": "running",
                "topics": self.topics,
                "subscription_name": self.subscription_name,
                "subscription_type": self.subscription_type,
                "consumer_name": self.consumer_name,
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


class PulsarDocumentConsumer(PulsarConsumer):
    """Specialized Pulsar consumer for document ingestion"""
    
    def __init__(
        self,
        service_url: str,
        subscription_name: str,
        document_processor: Callable,
        **kwargs
    ):
        super().__init__(
            service_url=service_url,
            subscription_name=subscription_name,
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


class PulsarStreamConsumer(PulsarConsumer):
    """Specialized Pulsar consumer for streaming data"""
    
    def __init__(
        self,
        service_url: str,
        subscription_name: str,
        stream_processor: Callable,
        topics: List[str],
        **kwargs
    ):
        super().__init__(
            service_url=service_url,
            subscription_name=subscription_name,
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
