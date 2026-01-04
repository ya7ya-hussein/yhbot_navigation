# isaaclab_bridge/socketio_client.py
"""
Socket.IO client for connecting Isaac Lab bridge to Express server.
Handles connection, disconnection, and reconnection logic.
"""

import socketio
import time
import threading
from typing import Optional, Callable, Dict, Any
from enum import Enum


class ConnectionStatus(Enum):
    """Connection status enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class SocketIOClient:
    """
    Socket.IO client for bridge-to-backend communication.
    Handles connection lifecycle and automatic reconnection.
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:3001",
        reconnection: bool = True,
        reconnection_delay: float = 1.0,
        reconnection_delay_max: float = 5.0,
        reconnection_attempts: int = 0
    ):
        """
        Initialize Socket.IO client.
        
        Args:
            server_url: Backend server URL
            reconnection: Enable automatic reconnection
            reconnection_delay: Initial reconnection delay in seconds
            reconnection_delay_max: Maximum reconnection delay
            reconnection_attempts: Max reconnection attempts (0 = infinite)
        """
        self._server_url = server_url
        self._reconnection = reconnection
        self._reconnection_delay = reconnection_delay
        self._reconnection_delay_max = reconnection_delay_max
        self._reconnection_attempts = reconnection_attempts
        
        self._sio = socketio.Client(
            reconnection=reconnection,
            reconnection_delay=reconnection_delay,
            reconnection_delay_max=reconnection_delay_max,
            reconnection_attempts=reconnection_attempts
        )
        
        self._status = ConnectionStatus.DISCONNECTED
        self._connection_error: Optional[str] = None
        self._status_lock = threading.Lock()
        
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup Socket.IO event handlers."""
        
        @self._sio.event
        def connect():
            """Handle successful connection."""
            with self._status_lock:
                self._status = ConnectionStatus.CONNECTED
                self._connection_error = None
            print(f"[Socket.IO] Connected to {self._server_url}")
        
        @self._sio.event
        def disconnect():
            """Handle disconnection."""
            with self._status_lock:
                self._status = ConnectionStatus.DISCONNECTED
            print(f"[Socket.IO] Disconnected from {self._server_url}")
        
        @self._sio.event
        def connect_error(data):
            """Handle connection error."""
            with self._status_lock:
                self._status = ConnectionStatus.ERROR
                self._connection_error = str(data)
            print(f"[Socket.IO] Connection error: {data}")
    
    def connect(self, timeout: float = 10.0) -> bool:
        """
        Connect to Socket.IO server.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            True if connected successfully
        """
        try:
            with self._status_lock:
                self._status = ConnectionStatus.CONNECTING
            
            self._sio.connect(
                self._server_url,
                wait_timeout=timeout
            )
            
            return self.is_connected()
            
        except Exception as e:
            with self._status_lock:
                self._status = ConnectionStatus.ERROR
                self._connection_error = str(e)
            print(f"[Socket.IO] Connection failed: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Socket.IO server."""
        try:
            if self.is_connected():
                self._sio.disconnect()
        except Exception as e:
            print(f"[Socket.IO] Disconnect error: {e}")
    
    def emit(
        self,
        event: str,
        data: Any = None,
        callback: Optional[Callable] = None
    ) -> bool:
        """
        Emit event to server.
        
        Args:
            event: Event name
            data: Event data
            callback: Optional callback function
            
        Returns:
            True if emitted successfully
        """
        try:
            if not self.is_connected():
                return False
            
            self._sio.emit(event, data, callback=callback)
            return True
            
        except Exception as e:
            print(f"[Socket.IO] Emit error: {e}")
            return False
    
    def on(self, event: str, handler: Callable) -> None:
        """
        Register event handler.
        
        Args:
            event: Event name
            handler: Handler function
        """
        self._sio.on(event, handler)
    
    def is_connected(self) -> bool:
        """
        Check if connected to server.
        
        Returns:
            True if connected
        """
        with self._status_lock:
            return self._status == ConnectionStatus.CONNECTED
    
    def get_status(self) -> ConnectionStatus:
        """
        Get current connection status.
        
        Returns:
            Connection status
        """
        with self._status_lock:
            return self._status
    
    def get_connection_error(self) -> Optional[str]:
        """
        Get last connection error.
        
        Returns:
            Error message or None
        """
        with self._status_lock:
            return self._connection_error
    
    def wait_for_connection(
        self,
        timeout: float = 10.0,
        check_interval: float = 0.1
    ) -> bool:
        """
        Wait for connection to be established.
        
        Args:
            timeout: Maximum wait time in seconds
            check_interval: Check interval in seconds
            
        Returns:
            True if connected within timeout
        """
        elapsed = 0.0
        
        while elapsed < timeout:
            if self.is_connected():
                return True
            
            time.sleep(check_interval)
            elapsed += check_interval
        
        return False
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.
        
        Returns:
            Dictionary with connection details
        """
        return {
            'server_url': self._server_url,
            'status': self.get_status().value,
            'connected': self.is_connected(),
            'error': self.get_connection_error(),
            'reconnection_enabled': self._reconnection,
            'sid': self._sio.sid if hasattr(self._sio, 'sid') else None
        }