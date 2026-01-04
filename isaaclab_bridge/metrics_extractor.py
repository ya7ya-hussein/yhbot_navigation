# isaaclab_bridge/metrics_extractor.py
"""
Metrics extraction functions for Isaac Lab environment.
Provides read-only access to environment metrics.
"""

from typing import Optional, Dict, Any, List
import torch
import numpy as np
from collections import deque


class MetricsExtractor:
    """
    Extracts metrics from Isaac Lab environment.
    Maintains session state for counters and aggregations.
    """
    
    def __init__(self, max_history_length: int = 100):
        self._success_count = 0
        self._collision_count = 0
        self._episode_times_history = deque(maxlen=max_history_length)
        self._total_episodes = 0
        # Cache for step_dt
        self._cached_step_dt: Optional[float] = None
    
    def reset_session(self) -> None:
        """Reset all session counters and history."""
        self._success_count = 0
        self._collision_count = 0
        self._episode_times_history.clear()
        self._total_episodes = 0
    
    def _get_robot(self, env: Any) -> Any:
        """
        Safely get robot articulation from environment scene.
        
        Args:
            env: Isaac Lab environment instance
            
        Returns:
            Robot articulation or None if not found
        """
        try:
            if not hasattr(env, 'scene'):
                return None
            
            scene = env.scene
            
            # Try direct dictionary access (Isaac Lab InteractiveScene supports this)
            try:
                robot = scene["robot"]
                if robot is not None:
                    return robot
            except (KeyError, TypeError):
                pass
            
            # Try attribute access
            if hasattr(scene, 'robot'):
                return scene.robot
            
            # Try articulations dict
            if hasattr(scene, 'articulations'):
                if 'robot' in scene.articulations:
                    return scene.articulations['robot']
            
            return None
            
        except Exception as e:
            print(f"[MetricsExtractor] Error getting robot: {e}")
            return None
    
    def extract_success_count(
        self,
        env: Any,
        env_index: int = 0
    ) -> Optional[int]:
        """
        Extract success counter from termination manager.
        
        Args:
            env: Isaac Lab environment instance
            env_index: Environment index to monitor
            
        Returns:
            Current success count or None on error
        """
        try:
            if not hasattr(env, 'termination_manager'):
                return None
            
            term_manager = env.termination_manager
            
            if not hasattr(term_manager, 'terminated'):
                return None
            
            terminated = term_manager.terminated
            
            if env_index >= terminated.shape[0]:
                return None
            
            is_terminated = terminated[env_index].item()
            
            if is_terminated:
                # Check if it's a success (goal reached)
                if hasattr(term_manager, '_term_dones'):
                    term_dones = term_manager._term_dones
                    
                    if 'goal_reached' in term_dones:
                        goal_reached = term_dones['goal_reached'][env_index].item()
                        if goal_reached:
                            self._success_count += 1
            
            return self._success_count
            
        except Exception:
            return None
    
    def extract_collision_count(
        self,
        env: Any,
        env_index: int = 0
    ) -> Optional[int]:
        """
        Extract collision counter from termination manager.
        
        Args:
            env: Isaac Lab environment instance
            env_index: Environment index to monitor
            
        Returns:
            Current collision count or None on error
        """
        try:
            if not hasattr(env, 'termination_manager'):
                return None
            
            term_manager = env.termination_manager
            
            if not hasattr(term_manager, 'terminated'):
                return None
            
            terminated = term_manager.terminated
            
            if env_index >= terminated.shape[0]:
                return None
            
            is_terminated = terminated[env_index].item()
            
            if is_terminated:
                # Check if it's a collision
                if hasattr(term_manager, '_term_dones'):
                    term_dones = term_manager._term_dones
                    
                    if 'collision' in term_dones:
                        collision = term_dones['collision'][env_index].item()
                        if collision:
                            self._collision_count += 1
            
            return self._collision_count
            
        except Exception:
            return None
    
    def get_step_dt(self, env: Any) -> float:
        """
        Get step dt from environment with caching.
        
        Args:
            env: Isaac Lab environment instance
            
        Returns:
            Step dt in seconds
        """
        if self._cached_step_dt is not None:
            return self._cached_step_dt
        
        try:
            if hasattr(env, 'step_dt'):
                self._cached_step_dt = float(env.step_dt)
                return self._cached_step_dt
            
            # Fallback: calculate from physics_dt and decimation
            if hasattr(env, 'physics_dt') and hasattr(env, 'cfg'):
                physics_dt = env.physics_dt
                decimation = getattr(env.cfg, 'decimation', 1)
                self._cached_step_dt = float(physics_dt * decimation)
                return self._cached_step_dt
            
            # Default fallback
            self._cached_step_dt = 1.0 / 15.0  # ~15 Hz default
            return self._cached_step_dt
            
        except Exception:
            return 1.0 / 15.0
    
    def extract_episode_time(
        self,
        env: Any,
        env_index: int = 0
    ) -> Optional[float]:
        """
        Extract current episode elapsed time.
        
        Args:
            env: Isaac Lab environment instance
            env_index: Environment index to monitor
            
        Returns:
            Episode time in seconds or None on error
        """
        try:
            if not hasattr(env, 'episode_length_buf'):
                return None
            
            episode_length_buf = env.episode_length_buf
            
            if env_index >= episode_length_buf.shape[0]:
                return None
            
            steps = episode_length_buf[env_index].item()
            step_dt = self.get_step_dt(env)
            elapsed_time = steps * step_dt
            
            return float(elapsed_time)
            
        except Exception:
            return None
    
    def calculate_episode_time_from_steps(
        self,
        steps: int,
        env: Any = None,
        step_dt: float = None
    ) -> float:
        """
        Calculate episode time from step count.
        
        Args:
            steps: Number of steps in episode
            env: Optional environment for getting step_dt
            step_dt: Optional explicit step_dt value
            
        Returns:
            Episode time in seconds
        """
        if step_dt is not None:
            return float(steps * step_dt)
        
        if env is not None:
            return float(steps * self.get_step_dt(env))
        
        # Fallback
        return float(steps * (1.0 / 15.0))
    
    def update_episode_history(
        self,
        env: Any = None,
        env_index: int = 0,
        episode_time: float = None,
        episode_steps: int = None
    ) -> None:
        """
        Update episode time history on episode end.
        
        Args:
            env: Isaac Lab environment instance
            env_index: Environment index to monitor
            episode_time: Direct episode time (optional)
            episode_steps: Episode steps to convert (optional)
        """
        try:
            # Priority: direct episode_time > episode_steps > extract from env
            if episode_time is not None:
                self._episode_times_history.append(episode_time)
                self._total_episodes += 1
                return
            
            if episode_steps is not None and env is not None:
                calculated_time = self.calculate_episode_time_from_steps(episode_steps, env)
                self._episode_times_history.append(calculated_time)
                self._total_episodes += 1
                return
            
            if env is not None:
                extracted_time = self.extract_episode_time(env, env_index)
                if extracted_time is not None:
                    self._episode_times_history.append(extracted_time)
                    self._total_episodes += 1
                
        except Exception:
            pass
    
    def extract_mean_episode_time(self) -> Optional[float]:
        """
        Calculate mean episode time from history.
        
        Returns:
            Mean episode time in seconds or None if no history
        """
        try:
            if len(self._episode_times_history) == 0:
                return None
            
            return float(np.mean(self._episode_times_history))
            
        except Exception:
            return None
    
    def extract_robot_velocity(
        self,
        env: Any,
        env_index: int = 0
    ) -> Optional[float]:
        """
        Extract robot linear velocity magnitude (2D).
        
        Args:
            env: Isaac Lab environment instance
            env_index: Environment index to monitor
            
        Returns:
            Velocity magnitude in m/s or None on error
        """
        try:
            robot = self._get_robot(env)
            
            if robot is None:
                return None
            
            if not hasattr(robot, 'data'):
                return None
            
            robot_data = robot.data
            
            # Try root_lin_vel_b first (body frame velocity)
            if hasattr(robot_data, 'root_lin_vel_b'):
                lin_vel = robot_data.root_lin_vel_b
                
                if env_index >= lin_vel.shape[0]:
                    return None
                
                # Get 2D velocity (x, y components)
                velocity_tensor = lin_vel[env_index, :2]
                velocity_magnitude = torch.norm(velocity_tensor).cpu().item()
                
                return float(velocity_magnitude)
            
            # Fallback: try root_lin_vel_w (world frame velocity)
            if hasattr(robot_data, 'root_lin_vel_w'):
                lin_vel = robot_data.root_lin_vel_w
                
                if env_index >= lin_vel.shape[0]:
                    return None
                
                velocity_tensor = lin_vel[env_index, :2]
                velocity_magnitude = torch.norm(velocity_tensor).cpu().item()
                
                return float(velocity_magnitude)
            
            return None
            
        except Exception as e:
            # Uncomment for debugging:
            # print(f"[MetricsExtractor] Velocity extraction error: {e}")
            return None
    
    def extract_robot_position(
        self,
        env: Any,
        env_index: int = 0
    ) -> Optional[Dict[str, float]]:
        """
        Extract robot position (x, y coordinates).
        
        Args:
            env: Isaac Lab environment instance
            env_index: Environment index to monitor
            
        Returns:
            Dictionary with x, y coordinates or None on error
        """
        try:
            robot = self._get_robot(env)
            
            if robot is None:
                return None
            
            if not hasattr(robot, 'data'):
                return None
            
            robot_data = robot.data
            
            if not hasattr(robot_data, 'root_pos_w'):
                return None
            
            position = robot_data.root_pos_w
            
            if env_index >= position.shape[0]:
                return None
            
            # Clone and move to CPU for safety
            pos_tensor = position[env_index, :2].clone().cpu()
            
            return {
                'x': float(pos_tensor[0].item()),
                'y': float(pos_tensor[1].item())
            }
            
        except Exception as e:
            # Uncomment for debugging:
            # print(f"[MetricsExtractor] Position extraction error: {e}")
            return None
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get current session statistics.
        
        Returns:
            Dictionary with session stats
        """
        return {
            'success_count': self._success_count,
            'collision_count': self._collision_count,
            'total_episodes': self._total_episodes,
            'episode_history_length': len(self._episode_times_history),
            'mean_episode_time': self.extract_mean_episode_time()
        }
    
    def detect_episode_reset(
        self,
        env: Any,
        env_index: int = 0
    ) -> bool:
        """
        Detect if episode just reset.
        
        Args:
            env: Isaac Lab environment instance
            env_index: Environment index to monitor
            
        Returns:
            True if episode just reset
        """
        try:
            if not hasattr(env, 'episode_length_buf'):
                return False
            
            current_step = env.episode_length_buf[env_index].item()
            
            # Episode reset detected if step counter is 0 or 1
            return current_step <= 1
            
        except Exception:
            return False