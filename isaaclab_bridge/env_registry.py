# isaaclab_bridge/env_registry.py
"""
Singleton registry for Isaac Lab environment instances.
Allows bridge to access running environment without modification.
"""

from typing import Optional, Any
import threading


class EnvironmentRegistry:
    """
    Singleton registry for storing active Isaac Lab environment instance.
    Thread-safe implementation.
    """
    
    _instance: Optional['EnvironmentRegistry'] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._environment: Optional[Any] = None
        self._env_lock: threading.Lock = threading.Lock()
        self._initialized = True
    
    def register(self, env: Any) -> None:
        """
        Register an environment instance.
        
        Args:
            env: Isaac Lab environment instance
        """
        with self._env_lock:
            self._environment = env
    
    def get(self) -> Optional[Any]:
        """
        Get the registered environment instance.
        
        Returns:
            Environment instance or None if not registered
        """
        with self._env_lock:
            return self._environment
    
    def clear(self) -> None:
        """Clear the registered environment."""
        with self._env_lock:
            self._environment = None
    
    def is_registered(self) -> bool:
        """Check if an environment is registered."""
        with self._env_lock:
            return self._environment is not None