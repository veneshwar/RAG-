"""
Window manager for streaming data processing
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import time
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class WindowItem:
    """Represents an item in a time window"""
    data: Any
    timestamp: float
    item_id: str


class TimeWindowManager:
    """Manages time-based windows for streaming data"""
    
    def __init__(
        self,
        window_size_seconds: int = 60,
        slide_interval_seconds: int = 10,
        max_windows: int = 100
    ):
        self.window_size_seconds = window_size_seconds
        self.slide_interval_seconds = slide_interval_seconds
        self.max_windows = max_windows
        
        self.windows: Dict[str, deque] = defaultdict(deque)
        self.window_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.is_running = False
        self._slide_task = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the window manager"""
        if self.is_running:
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        
        # Start sliding task
        self._slide_task = asyncio.create_task(self._slide_windows())
        
        logger.info(f"Time window manager started: {self.window_size_seconds}s windows, {self.slide_interval_seconds}s slide")
    
    async def stop(self):
        """Stop the window manager"""
        self.is_running = False
        self._shutdown_event.set()
        
        if self._slide_task:
            self._slide_task.cancel()
            try:
                await self._slide_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Time window manager stopped")
    
    async def add_item(self, window_key: str, data: Any, item_id: str = None) -> str:
        """
        Add an item to a time window
        """
        if not self.is_running:
            logger.warning("Window manager not running, item not added")
            return ""
        
        item_id = item_id or f"{window_key}_{int(time.time() * 1000000)}"
        
        item = WindowItem(
            data=data,
            timestamp=time.time(),
            item_id=item_id
        )
        
        self.windows[window_key].append(item)
        
        # Remove old items outside the window
        current_time = time.time()
        window_start = current_time - self.window_size_seconds
        
        while (self.windows[window_key] and 
               self.windows[window_key][0].timestamp < window_start):
            self.windows[window_key].popleft()
        
        logger.debug(f"Added item {item_id} to window {window_key}")
        return item_id
    
    def get_window_data(self, window_key: str) -> List[Any]:
        """
        Get all data in the current window
        """
        return [item.data for item in self.windows[window_key]]
    
    def get_window_items(self, window_key: str) -> List[WindowItem]:
        """
        Get all items in the current window
        """
        return list(self.windows[window_key])
    
    def get_window_count(self, window_key: str) -> int:
        """
        Get count of items in the current window
        """
        return len(self.windows[window_key])
    
    def get_window_stats(self, window_key: str) -> Dict[str, Any]:
        """
        Get statistics for a window
        """
        items = list(self.windows[window_key])
        
        if not items:
            return {
                "count": 0,
                "oldest_timestamp": None,
                "newest_timestamp": None,
                "time_span": 0.0
            }
        
        timestamps = [item.timestamp for item in items]
        
        return {
            "count": len(items),
            "oldest_timestamp": min(timestamps),
            "newest_timestamp": max(timestamps),
            "time_span": max(timestamps) - min(timestamps)
        }
    
    def add_window_handler(self, window_key: str, handler: Callable):
        """
        Add a handler for window slide events
        """
        self.window_handlers[window_key].append(handler)
        logger.info(f"Added handler for window {window_key}")
    
    def remove_window_handler(self, window_key: str, handler: Callable):
        """
        Remove a handler for window slide events
        """
        if handler in self.window_handlers[window_key]:
            self.window_handlers[window_key].remove(handler)
            logger.info(f"Removed handler for window {window_key}")
    
    async def _slide_windows(self):
        """
        Periodically slide windows and trigger handlers
        """
        while self.is_running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.slide_interval_seconds)
                
                # Trigger handlers for all windows
                for window_key in list(self.windows.keys()):
                    if self.windows[window_key]:  # Only if window has data
                        await self._trigger_window_handlers(window_key)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in window sliding: {str(e)}")
                await asyncio.sleep(1)
    
    async def _trigger_window_handlers(self, window_key: str):
        """
        Trigger all handlers for a window
        """
        window_data = self.get_window_data(window_key)
        window_stats = self.get_window_stats(window_key)
        
        for handler in self.window_handlers[window_key]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(window_key, window_data, window_stats)
                else:
                    handler(window_key, window_data, window_stats)
            except Exception as e:
                logger.error(f"Error in window handler: {str(e)}")
    
    def clear_window(self, window_key: str):
        """
        Clear all data from a window
        """
        if window_key in self.windows:
            self.windows[window_key].clear()
            logger.info(f"Cleared window {window_key}")
    
    def get_all_window_keys(self) -> List[str]:
        """
        Get all active window keys
        """
        return list(self.windows.keys())
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the window manager
        """
        stats = {
            "is_running": self.is_running,
            "window_size_seconds": self.window_size_seconds,
            "slide_interval_seconds": self.slide_interval_seconds,
            "active_windows": len(self.windows),
            "total_items": sum(len(window) for window in self.windows.values()),
            "window_details": {}
        }
        
        for window_key in self.windows:
            stats["window_details"][window_key] = self.get_window_stats(window_key)
        
        return stats


class CountWindowManager:
    """Manages count-based windows for streaming data"""
    
    def __init__(
        self,
        window_size: int = 100,
        slide_size: int = 20,
        max_windows: int = 100
    ):
        self.window_size = window_size
        self.slide_size = slide_size
        self.max_windows = max_windows
        
        self.windows: Dict[str, deque] = defaultdict(deque)
        self.window_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.item_counters: Dict[str, int] = defaultdict(int)
        self.is_running = False
    
    async def start(self):
        """Start the count window manager"""
        self.is_running = True
        logger.info(f"Count window manager started: {self.window_size} items per window, {self.slide_size} slide")
    
    async def stop(self):
        """Stop the count window manager"""
        self.is_running = False
        logger.info("Count window manager stopped")
    
    async def add_item(self, window_key: str, data: Any, item_id: str = None) -> str:
        """
        Add an item to a count window
        """
        if not self.is_running:
            logger.warning("Window manager not running, item not added")
            return ""
        
        item_id = item_id or f"{window_key}_{self.item_counters[window_key]}"
        self.item_counters[window_key] += 1
        
        item = WindowItem(
            data=data,
            timestamp=time.time(),
            item_id=item_id
        )
        
        self.windows[window_key].append(item)
        
        # Check if window is full
        if len(self.windows[window_key]) >= self.window_size:
            await self._trigger_window_handlers(window_key)
            
            # Slide the window
            for _ in range(self.slide_size):
                if self.windows[window_key]:
                    self.windows[window_key].popleft()
        
        return item_id
    
    def get_window_data(self, window_key: str) -> List[Any]:
        """
        Get all data in the current window
        """
        return [item.data for item in self.windows[window_key]]
    
    def get_window_items(self, window_key: str) -> List[WindowItem]:
        """
        Get all items in the current window
        """
        return list(self.windows[window_key])
    
    def get_window_count(self, window_key: str) -> int:
        """
        Get count of items in the current window
        """
        return len(self.windows[window_key])
    
    def add_window_handler(self, window_key: str, handler: Callable):
        """
        Add a handler for window events
        """
        self.window_handlers[window_key].append(handler)
        logger.info(f"Added handler for count window {window_key}")
    
    async def _trigger_window_handlers(self, window_key: str):
        """
        Trigger all handlers for a window
        """
        window_data = self.get_window_data(window_key)
        window_stats = {
            "count": len(window_data),
            "window_type": "count"
        }
        
        for handler in self.window_handlers[window_key]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(window_key, window_data, window_stats)
                else:
                    handler(window_key, window_data, window_stats)
            except Exception as e:
                logger.error(f"Error in count window handler: {str(e)}")
    
    def clear_window(self, window_key: str):
        """
        Clear all data from a window
        """
        if window_key in self.windows:
            self.windows[window_key].clear()
            logger.info(f"Cleared count window {window_key}")


class SessionWindowManager:
    """Manages session-based windows for streaming data"""
    
    def __init__(
        self,
        session_timeout_seconds: int = 1800,  # 30 minutes
        max_sessions: int = 1000
    ):
        self.session_timeout_seconds = session_timeout_seconds
        self.max_sessions = max_sessions
        
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_handlers: List[Callable] = []
        self.is_running = False
        self._cleanup_task = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """Start the session window manager"""
        if self.is_running:
            return
        
        self.is_running = True
        self._shutdown_event.clear()
        
        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        logger.info(f"Session window manager started: {self.session_timeout_seconds}s timeout")
    
    async def stop(self):
        """Stop the session window manager"""
        self.is_running = False
        self._shutdown_event.set()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Session window manager stopped")
    
    async def add_item(self, session_id: str, data: Any, item_id: str = None) -> str:
        """
        Add an item to a session window
        """
        if not self.is_running:
            logger.warning("Session manager not running, item not added")
            return ""
        
        current_time = time.time()
        item_id = item_id or f"{session_id}_{int(current_time * 1000000)}"
        
        # Initialize session if it doesn't exist
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "items": deque(),
                "created_at": current_time,
                "last_activity": current_time,
                "item_count": 0
            }
        
        session = self.sessions[session_id]
        
        # Add item to session
        item = WindowItem(
            data=data,
            timestamp=current_time,
            item_id=item_id
        )
        
        session["items"].append(item)
        session["last_activity"] = current_time
        session["item_count"] += 1
        
        # Limit session size
        if len(session["items"]) > 1000:  # Max 1000 items per session
            session["items"].popleft()
        
        # Check if we have too many sessions
        if len(self.sessions) > self.max_sessions:
            await self._cleanup_oldest_sessions()
        
        logger.debug(f"Added item {item_id} to session {session_id}")
        return item_id
    
    def get_session_data(self, session_id: str) -> List[Any]:
        """
        Get all data in a session
        """
        if session_id not in self.sessions:
            return []
        
        return [item.data for item in self.sessions[session_id]["items"]]
    
    def get_session_items(self, session_id: str) -> List[WindowItem]:
        """
        Get all items in a session
        """
        if session_id not in self.sessions:
            return []
        
        return list(self.sessions[session_id]["items"])
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """
        Get information about a session
        """
        if session_id not in self.sessions:
            return {}
        
        session = self.sessions[session_id]
        current_time = time.time()
        
        return {
            "session_id": session_id,
            "created_at": session["created_at"],
            "last_activity": session["last_activity"],
            "item_count": session["item_count"],
            "age_seconds": current_time - session["created_at"],
            "idle_seconds": current_time - session["last_activity"],
            "is_expired": (current_time - session["last_activity"]) > self.session_timeout_seconds
        }
    
    def add_session_handler(self, handler: Callable):
        """
        Add a handler for session events
        """
        self.session_handlers.append(handler)
        logger.info("Added session handler")
    
    async def _cleanup_expired_sessions(self):
        """
        Periodically clean up expired sessions
        """
        while self.is_running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if (current_time - session["last_activity"]) > self.session_timeout_seconds:
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    await self._remove_session(session_id, "expired")
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {str(e)}")
                await asyncio.sleep(1)
    
    async def _cleanup_oldest_sessions(self):
        """
        Remove oldest sessions when limit is exceeded
        """
        # Sort sessions by last activity
        sorted_sessions = sorted(
            self.sessions.items(),
            key=lambda x: x[1]["last_activity"]
        )
        
        # Remove oldest sessions
        to_remove = len(self.sessions) - self.max_sessions
        for i in range(to_remove):
            session_id = sorted_sessions[i][0]
            await self._remove_session(session_id, "limit_exceeded")
    
    async def _remove_session(self, session_id: str, reason: str):
        """
        Remove a session and trigger handlers
        """
        if session_id not in self.sessions:
            return
        
        session_data = self.get_session_data(session_id)
        session_info = self.get_session_info(session_id)
        
        # Trigger handlers
        for handler in self.session_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(session_id, session_data, session_info, reason)
                else:
                    handler(session_id, session_data, session_info, reason)
            except Exception as e:
                logger.error(f"Error in session handler: {str(e)}")
        
        # Remove session
        del self.sessions[session_id]
        logger.info(f"Removed session {session_id} ({reason})")
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all sessions
        """
        return {sid: self.get_session_info(sid) for sid in self.sessions.keys()}
    
    async def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the session manager
        """
        stats = {
            "is_running": self.is_running,
            "session_timeout_seconds": self.session_timeout_seconds,
            "max_sessions": self.max_sessions,
            "active_sessions": len(self.sessions),
            "total_items": sum(session["item_count"] for session in self.sessions.values())
        }
        
        return stats
