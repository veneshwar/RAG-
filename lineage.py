"""
Data lineage tracking for document processing
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import json


logger = logging.getLogger(__name__)


class ProcessingStage(str, Enum):
    """Processing stages for documents"""
    INGESTION = "ingestion"
    CLEANING = "cleaning"
    NORMALIZATION = "normalization"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    ENRICHMENT = "enrichment"


@dataclass
class LineageEvent:
    """Represents a lineage event"""
    event_id: str
    document_id: str
    stage: ProcessingStage
    timestamp: datetime
    status: str  # "started", "completed", "failed"
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class DataLineageTracker:
    """Tracks data lineage through processing pipeline"""
    
    def __init__(
        self,
        max_events: int = 10000,
        retention_days: int = 30
    ):
        self.max_events = max_events
        self.retention_days = retention_days
        
        self.events: Dict[str, LineageEvent] = {}
        self.document_events: Dict[str, List[str]] = {}  # document_id -> event_ids
        self.stage_events: Dict[ProcessingStage, List[str]] = {}  # stage -> event_ids
        
        self._lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            "total_events": 0,
            "events_by_stage": {},
            "events_by_status": {},
            "avg_processing_times": {}
        }
    
    async def start_processing(
        self,
        document_id: str,
        stage: ProcessingStage,
        input_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record the start of a processing stage
        """
        event_id = str(uuid.uuid4())
        
        event = LineageEvent(
            event_id=event_id,
            document_id=document_id,
            stage=stage,
            timestamp=datetime.utcnow(),
            status="started",
            input_data=input_data,
            metadata=metadata
        )
        
        await self._add_event(event)
        
        logger.debug(f"Started processing {document_id} at stage {stage}")
        return event_id
    
    async def complete_processing(
        self,
        event_id: str,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record the completion of a processing stage
        """
        try:
            async with self._lock:
                if event_id not in self.events:
                    return False
                
                event = self.events[event_id]
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - event.timestamp).total_seconds() * 1000
                
                # Update event
                event.status = "completed"
                event.output_data = output_data
                event.processing_time_ms = processing_time
                
                if metadata:
                    if event.metadata:
                        event.metadata.update(metadata)
                    else:
                        event.metadata = metadata
                
                # Update statistics
                await self._update_statistics(event)
                
                logger.debug(f"Completed processing {event.document_id} at stage {event.stage}")
                return True
                
        except Exception as e:
            logger.error(f"Error completing processing: {str(e)}")
            return False
    
    async def fail_processing(
        self,
        event_id: str,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Record the failure of a processing stage
        """
        try:
            async with self._lock:
                if event_id not in self.events:
                    return False
                
                event = self.events[event_id]
                
                # Calculate processing time
                processing_time = (datetime.utcnow() - event.timestamp).total_seconds() * 1000
                
                # Update event
                event.status = "failed"
                event.error_message = error_message
                event.processing_time_ms = processing_time
                
                if metadata:
                    if event.metadata:
                        event.metadata.update(metadata)
                    else:
                        event.metadata = metadata
                
                # Update statistics
                await self._update_statistics(event)
                
                logger.debug(f"Failed processing {event.document_id} at stage {event.stage}: {error_message}")
                return True
                
        except Exception as e:
            logger.error(f"Error recording processing failure: {str(e)}")
            return False
    
    async def get_document_lineage(self, document_id: str) -> List[LineageEvent]:
        """
        Get all events for a document
        """
        async with self._lock:
            if document_id not in self.document_events:
                return []
            
            event_ids = self.document_events[document_id]
            events = [self.events[eid] for eid in event_ids if eid in self.events]
            
            # Sort by timestamp
            events.sort(key=lambda e: e.timestamp)
            
            return events
    
    async def get_stage_events(
        self,
        stage: ProcessingStage,
        limit: int = 100
    ) -> List[LineageEvent]:
        """
        Get events for a specific stage
        """
        async with self._lock:
            if stage not in self.stage_events:
                return []
            
            event_ids = self.stage_events[stage][-limit:]  # Get most recent
            events = [self.events[eid] for eid in event_ids if eid in self.events]
            
            # Sort by timestamp (most recent first)
            events.sort(key=lambda e: e.timestamp, reverse=True)
            
            return events
    
    async def get_processing_chain(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get the complete processing chain for a document
        """
        events = await self.get_document_lineage(document_id)
        
        chain = []
        for event in events:
            chain.append({
                "stage": event.stage.value,
                "status": event.status,
                "timestamp": event.timestamp.isoformat(),
                "processing_time_ms": event.processing_time_ms,
                "error_message": event.error_message
            })
        
        return chain
    
    async def get_failed_documents(
        self,
        stage: Optional[ProcessingStage] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get documents that failed processing
        """
        async with self._lock:
            failed_events = []
            
            for event in self.events.values():
                if event.status == "failed":
                    if stage is None or event.stage == stage:
                        failed_events.append(event)
            
            # Sort by timestamp (most recent first)
            failed_events.sort(key=lambda e: e.timestamp, reverse=True)
            
            return [
                {
                    "document_id": event.document_id,
                    "stage": event.stage.value,
                    "error_message": event.error_message,
                    "timestamp": event.timestamp.isoformat()
                }
                for event in failed_events[:limit]
            ]
    
    async def get_processing_statistics(
        self,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get processing statistics
        """
        async with self._lock:
            cutoff_time = datetime.utcnow() - time_range if time_range else None
            
            filtered_events = []
            for event in self.events.values():
                if cutoff_time is None or event.timestamp >= cutoff_time:
                    filtered_events.append(event)
            
            # Calculate statistics
            stats = {
                "total_events": len(filtered_events),
                "events_by_stage": {},
                "events_by_status": {},
                "avg_processing_times": {},
                "success_rate": 0.0,
                "error_rate": 0.0
            }
            
            stage_times = {}
            
            for event in filtered_events:
                # Count by stage
                stage = event.stage.value
                stats["events_by_stage"][stage] = stats["events_by_stage"].get(stage, 0) + 1
                
                # Count by status
                status = event.status
                stats["events_by_status"][status] = stats["events_by_status"].get(status, 0) + 1
                
                # Track processing times
                if event.processing_time_ms is not None:
                    if stage not in stage_times:
                        stage_times[stage] = []
                    stage_times[stage].append(event.processing_time_ms)
            
            # Calculate averages
            for stage, times in stage_times.items():
                stats["avg_processing_times"][stage] = sum(times) / len(times)
            
            # Calculate rates
            total_completed = stats["events_by_status"].get("completed", 0)
            total_failed = stats["events_by_status"].get("failed", 0)
            total_processed = total_completed + total_failed
            
            if total_processed > 0:
                stats["success_rate"] = total_completed / total_processed
                stats["error_rate"] = total_failed / total_processed
            
            return stats
    
    async def cleanup_old_events(self):
        """
        Clean up old events beyond retention period
        """
        async with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(days=self.retention_days)
            
            # Find old events
            old_event_ids = []
            for event_id, event in self.events.items():
                if event.timestamp < cutoff_time:
                    old_event_ids.append(event_id)
            
            # Remove old events
            for event_id in old_event_ids:
                await self._remove_event(event_id)
            
            if old_event_ids:
                logger.info(f"Cleaned up {len(old_event_ids)} old lineage events")
    
    async def _add_event(self, event: LineageEvent):
        """
        Add an event to the tracker
        """
        async with self._lock:
            # Check if we need to remove old events
            if len(self.events) >= self.max_events:
                await self._remove_oldest_event()
            
            # Add event
            self.events[event.event_id] = event
            
            # Update document events
            if event.document_id not in self.document_events:
                self.document_events[event.document_id] = []
            self.document_events[event.document_id].append(event.event_id)
            
            # Update stage events
            if event.stage not in self.stage_events:
                self.stage_events[event.stage] = []
            self.stage_events[event.stage].append(event.event_id)
            
            # Update statistics
            self.stats["total_events"] = len(self.events)
    
    async def _remove_event(self, event_id: str):
        """
        Remove an event from the tracker
        """
        if event_id not in self.events:
            return
        
        event = self.events[event_id]
        
        # Remove from main storage
        del self.events[event_id]
        
        # Remove from document events
        if event.document_id in self.document_events:
            self.document_events[event.document_id].remove(event_id)
            if not self.document_events[event.document_id]:
                del self.document_events[event.document_id]
        
        # Remove from stage events
        if event.stage in self.stage_events:
            self.stage_events[event.stage].remove(event_id)
            if not self.stage_events[event.stage]:
                del self.stage_events[event.stage]
    
    async def _remove_oldest_event(self):
        """
        Remove the oldest event to make space
        """
        if not self.events:
            return
        
        oldest_event_id = min(
            self.events.keys(),
            key=lambda eid: self.events[eid].timestamp
        )
        
        await self._remove_event(oldest_event_id)
        logger.debug(f"Removed oldest lineage event {oldest_event_id}")
    
    async def _update_statistics(self, event: LineageEvent):
        """
        Update statistics based on an event
        """
        stage = event.stage.value
        status = event.status
        
        # Update counts
        self.stats["events_by_stage"][stage] = self.stats["events_by_stage"].get(stage, 0) + 1
        self.stats["events_by_status"][status] = self.stats["events_by_status"].get(status, 0) + 1
        
        # Update processing times
        if event.processing_time_ms is not None:
            if stage not in self.stats["avg_processing_times"]:
                self.stats["avg_processing_times"][stage] = []
            
            # Keep only recent times for average calculation
            times = self.stats["avg_processing_times"][stage]
            times.append(event.processing_time_ms)
            if len(times) > 100:  # Keep last 100 times
                self.stats["avg_processing_times"][stage] = times[-50:]
    
    async def export_lineage(
        self,
        document_id: str,
        format: str = "json"
    ) -> Optional[str]:
        """
        Export lineage data for a document
        """
        try:
            events = await self.get_document_lineage(document_id)
            
            if format == "json":
                lineage_data = {
                    "document_id": document_id,
                    "exported_at": datetime.utcnow().isoformat(),
                    "events": [asdict(event) for event in events]
                }
                
                # Convert datetime objects to strings
                for event in lineage_data["events"]:
                    event["timestamp"] = event["timestamp"].isoformat()
                    if event["metadata"]:
                        event["metadata"] = json.dumps(event["metadata"])
                
                return json.dumps(lineage_data, indent=2)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Error exporting lineage: {str(e)}")
            return None
    
    async def get_tracker_info(self) -> Dict[str, Any]:
        """
        Get information about the lineage tracker
        """
        async with self._lock:
            return {
                "total_events": len(self.events),
                "total_documents": len(self.document_events),
                "max_events": self.max_events,
                "retention_days": self.retention_days,
                "stages_tracked": [stage.value for stage in self.stage_events.keys()],
                "statistics": self.stats
            }
