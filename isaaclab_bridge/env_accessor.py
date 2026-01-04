# isaaclab_bridge/env_accessor.py
"""
Environment accessor module for reading Isaac Lab environment state.
Provides safe, read-only access to environment data.
"""

from typing import Optional, Any, Dict
import sys

from env_registry import EnvironmentRegistry


class EnvironmentAccessor:
    """
    Provides read-only access to Isaac Lab environment instance.
    Handles import errors and validates environment access.
    """
    
    def __init__(self):
        self._registry = EnvironmentRegistry()
        self._isaac_lab_available = False
        self._import_errors = []
        
        self._check_isaac_lab_imports()
    
    def _check_isaac_lab_imports(self) -> None:
        """Check if Isaac Lab modules are available."""
        required_modules = [
            'isaaclab',
            'isaaclab.envs',
            'isaaclab.assets',
            'isaaclab.managers',
            'isaaclab.sensors',
        ]
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError as e:
                self._import_errors.append({
                    'module': module,
                    'error': str(e)
                })
        
        self._isaac_lab_available = len(self._import_errors) == 0
    
    def get_environment(self) -> Optional[Any]:
        """
        Get the active Isaac Lab environment instance.
        
        Returns:
            Environment instance or None if not available
        """
        if not self._isaac_lab_available:
            return None
        
        return self._registry.get()
    
    def is_environment_available(self) -> bool:
        """
        Check if environment is available and accessible.
        
        Returns:
            True if environment is registered and accessible
        """
        return self._isaac_lab_available and self._registry.is_registered()
    
    def get_import_errors(self) -> list:
        """
        Get list of import errors encountered.
        
        Returns:
            List of import error dictionaries
        """
        return self._import_errors.copy()
    
    def verify_environment_access(self) -> Dict[str, Any]:
        """
        Verify read access to environment state.
        
        Returns:
            Dictionary with verification results
        """
        result = {
            'isaac_lab_available': self._isaac_lab_available,
            'environment_registered': self._registry.is_registered(),
            'import_errors': self._import_errors.copy(),
            'can_access_env': False,
            'env_properties': {}
        }
        
        if not self.is_environment_available():
            return result
        
        env = self.get_environment()
        
        try:
            # Verify read access to basic properties
            properties = {}
            
            if hasattr(env, 'num_envs'):
                properties['num_envs'] = env.num_envs
            
            if hasattr(env, 'device'):
                properties['device'] = str(env.device)
            
            if hasattr(env, 'physics_dt'):
                properties['physics_dt'] = env.physics_dt
            
            if hasattr(env, 'step_dt'):
                properties['step_dt'] = env.step_dt
            
            if hasattr(env, 'scene'):
                properties['has_scene'] = True
            
            if hasattr(env, 'episode_length_buf'):
                properties['has_episode_buffer'] = True
            
            result['can_access_env'] = True
            result['env_properties'] = properties
            
        except Exception as e:
            result['access_error'] = str(e)
        
        return result