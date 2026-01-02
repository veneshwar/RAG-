"""
Batch embedder for efficient processing of large document sets
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from app.indexing.embeddings.embedder import BaseEmbedder


logger = logging.getLogger(__name__)


class BatchEmbedder:
    """Efficient batch embedder with queuing and parallel processing"""
    
    def __init__(
        self,
        base_embedder: BaseEmbedder,
        max_queue_size: int = 1000,
        batch_size: int = 32,
        max_workers: int = 4,
        queue_timeout: float = 30.0,
        processing_timeout: float = 300.0,
        enable_progress_tracking: bool = True
    ):
        self.base_embedder = base_embedder
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.queue_timeout = queue_timeout
        self.processing_timeout = processing_timeout
        self.enable_progress_tracking = enable_progress_tracking
        
        self.embedding_queue = asyncio.Queue(maxsize=max_queue_size)
        self.result_futures = {}
        self.is_running = False
        self._workers = []
        self._shutdown_event = asyncio.Event()
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "total_failed": 0,
            "total_batches": 0,
            "avg_batch_time": 0.0,
            "queue_size": 0,
            "processing_rate": 0.0
        }
        
        self._batch_times = []
        self._start_time = None
    
    async def start(self):
        """Start the batch embedder workers"""
        if self.is_running:
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        self._start_time = time.time()
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"Batch embedder started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the batch embedder"""
        self.is_running = False
        self._shutdown_event.set()
        
        # Wait for all workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        logger.info("Batch embedder stopped")
    
    async def embed_text(self, text: str, priority: int = 0) -> List[float]:
        """
        Embed a single text with queuing
        """
        return await self.embed_batch([text], priority=priority)
    
    async def embed_batch(
        self,
        texts: List[str],
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> List[List[float]]:
        """
        Embed multiple texts with queuing
        """
        if not texts:
            return []
        
        # Create unique request ID
        request_id = f"req_{int(time.time() * 1000000)}_{id(texts)}"
        
        # Create future for result
        result_future = asyncio.Future()
        self.result_futures[request_id] = result_future
        
        # Add to queue
        queue_item = {
            "request_id": request_id,
            "texts": texts,
            "priority": priority,
            "timestamp": time.time()
        }
        
        try:
            await asyncio.wait_for(
                self.embedding_queue.put(queue_item),
                timeout=self.queue_timeout
            )
        except asyncio.TimeoutError:
            # Remove future and raise timeout
            self.result_futures.pop(request_id, None)
            raise TimeoutError(f"Queue full, timeout after {self.queue_timeout}s")
        
        # Wait for result
        timeout = timeout or self.processing_timeout
        try:
            result = await asyncio.wait_for(result_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.result_futures.pop(request_id, None)
            raise TimeoutError(f"Processing timeout after {timeout}s")
    
    async def embed_documents(
        self,
        documents: List[Dict[str, Any]],
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Embed documents with metadata preservation
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc.get("content", "") for doc in documents]
        
        # Get embeddings
        embeddings = await self.embed_batch(texts, priority=priority, timeout=timeout)
        
        # Add embeddings to documents
        embedded_documents = []
        for doc, embedding in zip(documents, embeddings):
            embedded_doc = doc.copy()
            embedded_doc["embedding"] = embedding
            embedded_doc["embedding_metadata"] = {
                "model_name": self.base_embedder.model_name,
                "embedding_dim": self.base_embedder.embedding_dim,
                "batch_embedder": True
            }
            embedded_documents.append(embedded_doc)
        
        return embedded_documents
    
    async def _worker(self, worker_name: str):
        """
        Worker task for processing embedding requests
        """
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running and not self._shutdown_event.is_set():
            try:
                # Get batch from queue
                batch_items = await self._get_batch()
                
                if not batch_items:
                    await asyncio.sleep(0.1)
                    continue
                
                # Process batch
                await self._process_batch(batch_items, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {str(e)}")
                await asyncio.sleep(1)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _get_batch(self) -> List[Dict[str, Any]]:
        """
        Get a batch of items from the queue
        """
        batch_items = []
        
        try:
            # Get first item
            first_item = await asyncio.wait_for(
                self.embedding_queue.get(),
                timeout=1.0
            )
            batch_items.append(first_item)
            
            # Try to get more items for batching
            while len(batch_items) < self.batch_size:
                try:
                    item = await asyncio.wait_for(
                        self.embedding_queue.get(),
                        timeout=0.1
                    )
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break
            
        except asyncio.TimeoutError:
            # No items available
            pass
        
        return batch_items
    
    async def _process_batch(self, batch_items: List[Dict[str, Any]], worker_name: str):
        """
        Process a batch of embedding requests
        """
        batch_start_time = time.time()
        
        try:
            # Collect all texts from batch
            all_texts = []
            text_mapping = []  # Maps text index to (request_id, text_index)
            
            for item in batch_items:
                request_id = item["request_id"]
                texts = item["texts"]
                
                for text_idx, text in enumerate(texts):
                    all_texts.append(text)
                    text_mapping.append((request_id, text_idx))
            
            # Generate embeddings for all texts
            embeddings = await self.base_embedder.embed_batch(all_texts)
            
            # Distribute results back to requests
            request_results = {}
            
            for (request_id, text_idx), embedding in zip(text_mapping, embeddings):
                if request_id not in request_results:
                    request_results[request_id] = []
                request_results[request_id].append((text_idx, embedding))
            
            # Complete futures
            for request_id, results in request_results.items():
                if request_id in self.result_futures:
                    # Sort results by text index
                    results.sort(key=lambda x: x[0])
                    final_embeddings = [emb for _, emb in results]
                    
                    future = self.result_futures.pop(request_id)
                    if not future.done():
                        future.set_result(final_embeddings)
            
            # Update statistics
            batch_time = time.time() - batch_start_time
            self._batch_times.append(batch_time)
            
            # Keep only recent batch times
            if len(self._batch_times) > 100:
                self._batch_times = self._batch_times[-50:]
            
            self.stats["total_processed"] += len(all_texts)
            self.stats["total_batches"] += 1
            self.stats["avg_batch_time"] = np.mean(self._batch_times)
            
            if self._start_time:
                elapsed = time.time() - self._start_time
                self.stats["processing_rate"] = self.stats["total_processed"] / elapsed
            
            logger.debug(f"{worker_name} processed batch of {len(all_texts)} texts in {batch_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch in {worker_name}: {str(e)}")
            
            # Fail all requests in batch
            for item in batch_items:
                request_id = item["request_id"]
                if request_id in self.result_futures:
                    future = self.result_futures.pop(request_id)
                    if not future.done():
                        future.set_exception(e)
            
            self.stats["total_failed"] += sum(len(item["texts"]) for item in batch_items)
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status
        """
        return {
            "queue_size": self.embedding_queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "active_futures": len(self.result_futures),
            "is_running": self.is_running,
            "workers": len(self._workers)
        }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics
        """
        queue_status = await self.get_queue_status()
        
        return {
            **self.stats,
            **queue_status,
            "success_rate": (
                self.stats["total_processed"] / 
                (self.stats["total_processed"] + self.stats["total_failed"])
                if (self.stats["total_processed"] + self.stats["total_failed"]) > 0 else 0.0
            )
        }
    
    async def clear_queue(self):
        """
        Clear the embedding queue
        """
        # Cancel all pending futures
        for request_id, future in self.result_futures.items():
            if not future.done():
                future.cancel()
        
        self.result_futures.clear()
        
        # Clear queue
        while not self.embedding_queue.empty():
            try:
                self.embedding_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info("Embedding queue cleared")


class PriorityBatchEmbedder(BatchEmbedder):
    """Batch embedder with priority queuing"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.priority_queue = asyncio.PriorityQueue(maxsize=self.max_queue_size)
    
    async def embed_batch(
        self,
        texts: List[str],
        priority: int = 0,
        timeout: Optional[float] = None
    ) -> List[List[float]]:
        """
        Embed texts with priority queuing
        """
        if not texts:
            return []
        
        # Create unique request ID
        request_id = f"req_{int(time.time() * 1000000)}_{id(texts)}"
        
        # Create future for result
        result_future = asyncio.Future()
        self.result_futures[request_id] = result_future
        
        # Add to priority queue (negative priority for max-heap behavior)
        queue_item = (-priority, request_id, texts, time.time())
        
        try:
            await asyncio.wait_for(
                self.priority_queue.put(queue_item),
                timeout=self.queue_timeout
            )
        except asyncio.TimeoutError:
            self.result_futures.pop(request_id, None)
            raise TimeoutError(f"Queue full, timeout after {self.queue_timeout}s")
        
        # Wait for result
        timeout = timeout or self.processing_timeout
        try:
            result = await asyncio.wait_for(result_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            self.result_futures.pop(request_id, None)
            raise TimeoutError(f"Processing timeout after {timeout}s")
    
    async def _get_batch(self) -> List[Dict[str, Any]]:
        """
        Get a batch of items from the priority queue
        """
        batch_items = []
        
        try:
            # Get first item
            priority, request_id, texts, timestamp = await asyncio.wait_for(
                self.priority_queue.get(),
                timeout=1.0
            )
            
            batch_items.append({
                "request_id": request_id,
                "texts": texts,
                "priority": -priority,  # Convert back to positive
                "timestamp": timestamp
            })
            
            # Try to get more items for batching
            while len(batch_items) < self.batch_size:
                try:
                    priority, request_id, texts, timestamp = await asyncio.wait_for(
                        self.priority_queue.get(),
                        timeout=0.1
                    )
                    
                    batch_items.append({
                        "request_id": request_id,
                        "texts": texts,
                        "priority": -priority,
                        "timestamp": timestamp
                    })
                except asyncio.TimeoutError:
                    break
            
        except asyncio.TimeoutError:
            pass
        
        return batch_items


class StreamingBatchEmbedder(BatchEmbedder):
    """Batch embedder optimized for streaming data"""
    
    def __init__(
        self,
        stream_window_size: int = 100,
        stream_timeout: float = 5.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.stream_window_size = stream_window_size
        self.stream_timeout = stream_timeout
        self.stream_buffer = []
        self.stream_buffer_lock = asyncio.Lock()
    
    async def add_stream_text(self, text: str, priority: int = 0) -> str:
        """
        Add text to streaming buffer
        """
        text_id = f"stream_{int(time.time() * 1000000)}_{id(text)}"
        
        async with self.stream_buffer_lock:
            self.stream_buffer.append({
                "text_id": text_id,
                "text": text,
                "priority": priority,
                "timestamp": time.time()
            })
            
            # Process buffer if full
            if len(self.stream_buffer) >= self.stream_window_size:
                await self._process_stream_buffer()
        
        return text_id
    
    async def flush_stream_buffer(self):
        """
        Process remaining items in stream buffer
        """
        async with self.stream_buffer_lock:
            if self.stream_buffer:
                await self._process_stream_buffer()
    
    async def _process_stream_buffer(self):
        """
        Process the stream buffer
        """
        if not self.stream_buffer:
            return
        
        # Extract texts
        texts = [item["text"] for item in self.stream_buffer]
        
        # Embed texts
        embeddings = await self.embed_batch(texts)
        
        # Store results (could be sent to callback, stored in database, etc.)
        for i, item in enumerate(self.stream_buffer):
            item["embedding"] = embeddings[i]
            # Here you could process the embedded item
        
        # Clear buffer
        self.stream_buffer.clear()
    
    async def start_stream_processor(self):
        """
        Start periodic stream processor
        """
        async def stream_processor():
            while self.is_running:
                await asyncio.sleep(self.stream_timeout)
                await self.flush_stream_buffer()
        
        asyncio.create_task(stream_processor())
