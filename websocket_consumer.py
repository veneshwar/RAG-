"""
WebSocket consumer for real-time data ingestion
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Optional, List, Set
import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed


logger = logging.getLogger(__name__)


class WebSocketConsumer:
    """WebSocket consumer for real-time data processing"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        max_connections: int = 100,
        ping_interval: int = 20,
        ping_timeout: int = 20,
        close_timeout: int = 10
    ):
        self.host = host
        self.port = port
        self.max_connections = max_connections
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.close_timeout = close_timeout
        
        self.server = None
        self.connections: Set[WebSocketServerProtocol] = set()
        self.message_handlers = {}
        self.is_running = False
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the WebSocket server"""
        try:
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                max_size=10**7,  # 10MB max message size
                max_queue=self.max_connections,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout,
                close_timeout=self.close_timeout
            )
            
            self.is_running = True
            
            logger.info(f"WebSocket server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {str(e)}")
            raise
    
    async def stop(self):
        """Stop the WebSocket server"""
        self.is_running = False
        self._shutdown_event.set()
        
        # Close all connections
        if self.connections:
            await asyncio.gather(
                *[connection.close() for connection in self.connections],
                return_exceptions=True
            )
        
        # Close server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("WebSocket server stopped")
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new WebSocket connection"""
        try:
            # Check connection limit
            if len(self.connections) >= self.max_connections:
                logger.warning(f"Connection limit reached, rejecting connection from {websocket.remote_address}")
                await websocket.close(code=1013, reason="Server overloaded")
                return
            
            # Add connection to set
            self.connections.add(websocket)
            logger.info(f"New connection from {websocket.remote_address}, total connections: {len(self.connections)}")
            
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "welcome",
                "message": "Connected to real-time RAG system",
                "timestamp": asyncio.get_event_loop().time()
            }))
            
            # Handle messages from this connection
            await self._handle_messages(websocket)
            
        except ConnectionClosed:
            logger.info(f"Connection closed by client: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error handling connection {websocket.remote_address}: {str(e)}")
        finally:
            # Remove connection from set
            self.connections.discard(websocket)
            logger.info(f"Connection removed, total connections: {len(self.connections)}")
    
    async def _handle_messages(self, websocket: WebSocketServerProtocol):
        """Handle messages from a WebSocket connection"""
        try:
            async for message in websocket:
                try:
                    # Parse JSON message
                    data = json.loads(message)
                    await self._process_message(websocket, data)
                
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": asyncio.get_event_loop().time()
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "Error processing message",
                        "timestamp": asyncio.get_event_loop().time()
                    }))
        
        except ConnectionClosed:
            pass  # Normal closure
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
    
    async def _process_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Process a single message"""
        try:
            message_type = data.get("type")
            
            if not message_type:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Missing message type",
                    "timestamp": asyncio.get_event_loop().time()
                }))
                return
            
            # Get handler for this message type
            handler = self.message_handlers.get(message_type)
            if handler:
                try:
                    response = await handler(websocket, data)
                    if response:
                        await websocket.send(json.dumps({
                            "type": "response",
                            "data": response,
                            "timestamp": asyncio.get_event_loop().time()
                        }))
                except Exception as e:
                    logger.error(f"Error in handler for type {message_type}: {str(e)}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Error processing {message_type}",
                        "timestamp": asyncio.get_event_loop().time()
                    }))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}",
                    "timestamp": asyncio.get_event_loop().time()
                }))
        
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler for a specific type"""
        self.message_handlers[message_type] = handler
        logger.info(f"Registered handler for message type: {message_type}")
    
    def unregister_handler(self, message_type: str):
        """Unregister a message handler"""
        if message_type in self.message_handlers:
            del self.message_handlers[message_type]
            logger.info(f"Unregistered handler for message type: {message_type}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.connections:
            return
        
        message_str = json.dumps(message)
        disconnected = set()
        
        for websocket in self.connections:
            try:
                await websocket.send(message_str)
            except ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to {websocket.remote_address}: {str(e)}")
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.connections -= disconnected
    
    async def send_to_connection(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """Send message to specific connection"""
        try:
            await websocket.send(json.dumps(message))
        except ConnectionClosed:
            self.connections.discard(websocket)
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
    
    async def get_connection_info(self) -> Dict[str, Any]:
        """Get connection information"""
        return {
            "status": "running" if self.is_running else "stopped",
            "host": self.host,
            "port": self.port,
            "connections": len(self.connections),
            "max_connections": self.max_connections,
            "handlers": list(self.message_handlers.keys()),
            "client_addresses": [str(ws.remote_address) for ws in self.connections]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the WebSocket consumer"""
        if not self.is_running:
            return {"status": "unhealthy", "reason": "server not running"}
        
        try:
            info = await self.get_connection_info()
            return {
                "status": "healthy",
                "info": info
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class DocumentWebSocketConsumer(WebSocketConsumer):
    """Specialized WebSocket consumer for document ingestion"""
    
    def __init__(self, document_processor: Callable, **kwargs):
        super().__init__(**kwargs)
        self.document_processor = document_processor
        
        # Register handlers
        self.register_handler("document_ingest", self._handle_document_ingest)
        self.register_handler("batch_ingest", self._handle_batch_ingest)
    
    async def _handle_document_ingest(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle single document ingestion"""
        try:
            document_id = data.get("document_id")
            content = data.get("content")
            metadata = data.get("metadata", {})
            source = data.get("source", "websocket")
            
            if not document_id or not content:
                return {"error": "Missing document_id or content"}
            
            logger.info(f"Processing document {document_id} from WebSocket")
            
            # Process the document
            result = await self.document_processor(
                document_id=document_id,
                content=content,
                metadata=metadata,
                source=source
            )
            
            return {
                "document_id": document_id,
                "status": "processed",
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": str(e)}
    
    async def _handle_batch_ingest(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle batch document ingestion"""
        try:
            documents = data.get("documents", [])
            results = []
            
            for doc in documents:
                try:
                    result = await self._handle_document_ingest(websocket, doc)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
            
            return {
                "batch_id": data.get("batch_id"),
                "processed": len([r for r in results if "error" not in r]),
                "failed": len([r for r in results if "error" in r]),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            return {"error": str(e)}
