# isaaclab_bridge/session_manager.py
"""
Session management for Isaac Lab Bridge.
Handles state, counters, and episode history.
"""

import json
import time
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime
from collections import deque
from pathlib import Path


class SessionManager:
    """
    Manages session state, counters, and episode history.
    Handles initialization, resets, and persistence.
    """
    
    def __init__(
        self,
        max_episode_history: int = 1000,
        persist_data: bool = False,
        persistence_dir: Optional[str] = None
    ):
        """
        Initialize session manager.
        
        Args:
            max_episode_history: Maximum episodes to keep in history
            persist_data: Enable data persistence to disk
            persistence_dir: Directory for persisted data
        """
        self._max_episode_history = max_episode_history
        self._persist_data = persist_data
        self._persistence_dir = Path(persistence_dir) if persistence_dir else Path("./session_data")
        
        self._session_id: Optional[str] = None
        self._session_start_time: Optional[float] = None
        self._session_active = False
        
        self._counters = {
            'success': 0,
            'collision': 0,
            'timeout': 0,
            'total_episodes': 0,
            'total_steps': 0
        }
        
        self._episode_history: deque = deque(maxlen=max_episode_history)
        
        self._current_episode = {
            'episode_id': 0,
            'start_time': None,
            'start_step': 0,
            'active': False
        }
        
        self._lock = threading.Lock()
        
        if self._persist_data:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start new session.
        
        Args:
            session_id: Optional session ID, auto-generated if None
            
        Returns:
            Session ID
        """
        with self._lock:
            if session_id is None:
                session_id = self._generate_session_id()
            
            self._session_id = session_id
            self._session_start_time = time.time()
            self._session_active = True
            
            self._reset_all_counters()
            self._episode_history.clear()
            self._current_episode = {
                'episode_id': 0,
                'start_time': None,
                'start_step': 0,
                'active': False
            }
            
            print(f"[SessionManager] Session started: {session_id}")
            
            return session_id
    
    def stop_session(self) -> None:
        """Stop current session."""
        with self._lock:
            if not self._session_active:
                return
            
            session_duration = time.time() - self._session_start_time if self._session_start_time else 0
            
            print(f"[SessionManager] Session stopped: {self._session_id}")
            print(f"  Duration: {session_duration:.2f}s")
            print(f"  Episodes: {self._counters['total_episodes']}")
            print(f"  Success: {self._counters['success']}")
            print(f"  Collisions: {self._counters['collision']}")
            
            self._session_active = False
            
            # Persist after releasing lock to avoid blocking
            if self._persist_data:
                try:
                    self._persist_session_data_unlocked()
                except Exception as e:
                    print(f"[SessionManager] Persistence error: {e}")

    def _persist_session_data_unlocked(self) -> None:
        """Persist session data to disk (call without lock held)."""
        if not self._session_id:
            return
        
        try:
            data = {
                'session_id': self._session_id,
                'start_time': self._session_start_time,
                'end_time': time.time(),
                'counters': self._counters.copy(),
                'episode_history': list(self._episode_history),
                'statistics': {
                    'total_episodes': self._counters['total_episodes'],
                    'success_count': self._counters['success'],
                    'collision_count': self._counters['collision']
                }
            }
            
            filepath = self._persistence_dir / f"{self._session_id}.json"
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[SessionManager] Session data persisted: {filepath}")
            
        except Exception as e:
            print(f"[SessionManager] Persistence error: {e}")
    
    def _persist_session_data(self) -> None:
        """Persist session data to disk (legacy method with lock)."""
        # For backward compatibility - just call unlocked version
        self._persist_session_data_unlocked()
    
    def reset_counters(self) -> None:
        """Reset all session counters."""
        with self._lock:
            self._reset_all_counters()
            print("[SessionManager] Counters reset")
    
    def _reset_all_counters(self) -> None:
        """Internal counter reset (no lock)."""
        self._counters = {
            'success': 0,
            'collision': 0,
            'timeout': 0,
            'total_episodes': 0,
            'total_steps': 0
        }
    
    def increment_counter(self, counter_name: str, amount: int = 1) -> None:
        """
        Increment specific counter.
        
        Args:
            counter_name: Counter to increment
            amount: Amount to increment
        """
        with self._lock:
            if counter_name in self._counters:
                self._counters[counter_name] += amount
    
    def get_counter(self, counter_name: str) -> int:
        """
        Get counter value.
        
        Args:
            counter_name: Counter name
            
        Returns:
            Counter value
        """
        with self._lock:
            return self._counters.get(counter_name, 0)
    
    def start_episode(self, global_step: int = 0) -> int:
        """
        Start new episode.
        
        Args:
            global_step: Global step counter
            
        Returns:
            Episode ID
        """
        with self._lock:
            episode_id = self._counters['total_episodes']
            
            self._current_episode = {
                'episode_id': episode_id,
                'start_time': time.time(),
                'start_step': global_step,
                'active': True,
                'data': {}
            }
            
            return episode_id
    
    def end_episode(
        self,
        global_step: int,
        termination_reason: str,
        final_reward: float = 0.0,
        success: bool = False,
        **kwargs
    ) -> None:
        """
        End current episode and save to history.
        
        Args:
            global_step: Global step counter
            termination_reason: Reason for termination
            final_reward: Final episode reward
            success: Whether episode succeeded
            **kwargs: Additional episode data
        """
        with self._lock:
            if not self._current_episode['active']:
                return
            
            end_time = time.time()
            duration = end_time - self._current_episode['start_time']
            steps = global_step - self._current_episode['start_step']
            
            episode_data = {
                'episode_id': self._current_episode['episode_id'],
                'start_time': self._current_episode['start_time'],
                'end_time': end_time,
                'duration': duration,
                'steps': steps,
                'start_step': self._current_episode['start_step'],
                'end_step': global_step,
                'termination_reason': termination_reason,
                'final_reward': final_reward,
                'success': success,
                **kwargs
            }
            
            self._episode_history.append(episode_data)
            
            self._counters['total_episodes'] += 1
            self._counters['total_steps'] += steps
            
            if termination_reason == 'success':
                self._counters['success'] += 1
            elif termination_reason == 'collision':
                self._counters['collision'] += 1
            elif termination_reason == 'timeout':
                self._counters['timeout'] += 1
            
            self._current_episode['active'] = False
    
    def get_episode_history(
        self,
        count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get episode history.
        
        Args:
            count: Number of recent episodes (None = all)
            
        Returns:
            List of episode data
        """
        with self._lock:
            if count is None:
                return list(self._episode_history)
            else:
                return list(self._episode_history)[-count:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_episodes = self._counters['total_episodes']
            
            stats = {
                'session_id': self._session_id,
                'session_active': self._session_active,
                'session_duration': time.time() - self._session_start_time if self._session_start_time else 0,
                'counters': self._counters.copy(),
                'success_rate': self._counters['success'] / total_episodes if total_episodes > 0 else 0,
                'collision_rate': self._counters['collision'] / total_episodes if total_episodes > 0 else 0,
                'timeout_rate': self._counters['timeout'] / total_episodes if total_episodes > 0 else 0,
                'episode_history_length': len(self._episode_history),
                'current_episode_active': self._current_episode['active']
            }
            
            if len(self._episode_history) > 0:
                recent_episodes = list(self._episode_history)[-100:]
                
                stats['recent_metrics'] = {
                    'avg_duration': sum(ep['duration'] for ep in recent_episodes) / len(recent_episodes),
                    'avg_steps': sum(ep['steps'] for ep in recent_episodes) / len(recent_episodes),
                    'avg_reward': sum(ep.get('final_reward', 0) for ep in recent_episodes) / len(recent_episodes),
                    'recent_success_rate': sum(1 for ep in recent_episodes if ep['success']) / len(recent_episodes)
                }
            
            return stats
    
    def clear_history(self) -> None:
        """Clear episode history."""
        with self._lock:
            self._episode_history.clear()
            print("[SessionManager] Episode history cleared")
    
    def on_dashboard_connect(self) -> None:
        """Handle dashboard connection."""
        print("[SessionManager] Dashboard connected - counters maintained")
    
    def on_dashboard_disconnect(self) -> None:
        """Handle dashboard disconnection."""
        print("[SessionManager] Dashboard disconnected - counters maintained")
    
    def on_environment_restart(self) -> None:
        """Handle environment restart."""
        with self._lock:
            print("[SessionManager] Environment restart detected")
            
            if self._persist_data:
                self._persist_session_data()
            
            self._reset_all_counters()
            self._episode_history.clear()
            
            self._current_episode = {
                'episode_id': 0,
                'start_time': None,
                'start_step': 0,
                'active': False
            }
            
            print("[SessionManager] State cleared for restart")
    
    def _persist_session_data(self) -> None:
        """Persist session data to disk."""
        if not self._session_id:
            return
        
        try:
            data = {
                'session_id': self._session_id,
                'start_time': self._session_start_time,
                'end_time': time.time(),
                'counters': self._counters.copy(),
                'episode_history': list(self._episode_history),
                'statistics': self.get_statistics()
            }
            
            filepath = self._persistence_dir / f"{self._session_id}.json"
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"[SessionManager] Session data persisted: {filepath}")
            
        except Exception as e:
            print(f"[SessionManager] Persistence error: {e}")
    
    def load_session(self, session_id: str) -> bool:
        """
        Load session from disk.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if loaded successfully
        """
        try:
            filepath = self._persistence_dir / f"{session_id}.json"
            
            if not filepath.exists():
                return False
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            with self._lock:
                self._session_id = data['session_id']
                self._session_start_time = data['start_time']
                self._counters = data['counters']
                self._episode_history = deque(data['episode_history'], maxlen=self._max_episode_history)
                self._session_active = True
            
            print(f"[SessionManager] Session loaded: {session_id}")
            return True
            
        except Exception as e:
            print(f"[SessionManager] Load error: {e}")
            return False
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"session_{timestamp}"
    
    def is_active(self) -> bool:
        """Check if session is active."""
        with self._lock:
            return self._session_active
    
    def get_session_id(self) -> Optional[str]:
        """Get current session ID."""
        with self._lock:
            return self._session_id