# isaaclab_bridge/data_emitter.py
"""
Data emission loop for streaming metrics and frames to backend server.
Runs alongside Isaac Lab and emits real-time data.

IMPORTANT: Frame capture must happen on the MAIN THREAD. 
Use submit_frame() to pass frames captured from the main simulation loop.
"""

import time
import threading
from typing import Optional, Dict, Any
from datetime import datetime

from socketio_client import SocketIOClient, ConnectionStatus
from metrics_extractor import MetricsExtractor
from env_accessor import EnvironmentAccessor
from session_manager import SessionManager


class DataEmitter:
    """
    Real-time data emission loop for Isaac Lab metrics and frames.
    Streams to backend server via Socket.IO.
    
    Frame capture must happen externally on the main thread.
    Call submit_frame() to queue frames for emission.
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:3001",
        target_hz: float = 30.0,
        env_index: int = 0,
        auto_reconnect: bool = True,
        session_manager: Optional[SessionManager] = None

    ):
        """
        Initialize data emitter.
        
        Args:
            server_url: Backend server URL
            target_hz: Target emission frequency for metrics
            env_index: Environment index to monitor
            auto_reconnect: Enable automatic reconnection
            session_manager: Session manager instance
        """
        self._server_url = server_url
        self._target_hz = target_hz
        self._target_period = 1.0 / target_hz
        self._env_index = env_index
        
        self._socketio_client = SocketIOClient(
            server_url=server_url,
            reconnection=auto_reconnect
        )
        self._metrics_extractor = MetricsExtractor()
        self._env_accessor = EnvironmentAccessor()
        self._session_manager = session_manager or SessionManager()
        
        self._running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()
        
        # Frame queue (set from main thread, read from emitter thread)
        self._pending_frame: Optional[str] = None
        self._pending_frame_timestamp: float = 0.0
        self._frame_lock = threading.Lock()
        
        self._emission_count = 0
        self._frame_emission_count = 0
        self._error_count = 0
        self._last_emission_time = 0.0
        
        # Episode tracking state
        self._prev_episode_length: int = 0
        self._tracking_initialized: bool = False
        self._reset_threshold: int = 5
        
        # Accumulated termination signals (latching)
        self._accumulated_termination: Dict[str, bool] = {
            'goal_reached': False,
            'collision': False,
            'time_out': False
        }
        
        # Debug flag
        self._debug_extraction = True
        self._debug_logged_once = False
        
        self._setup_socketio_handlers()

    def _setup_socketio_handlers(self) -> None:
        """Setup Socket.IO event handlers."""
        
        @self._socketio_client._sio.event
        def dashboard_connected(data):
            """Handle dashboard connection."""
            print("[DataEmitter] Dashboard connected")
            self._session_manager.on_dashboard_connect()
        
        @self._socketio_client._sio.event
        def dashboard_disconnected(data):
            """Handle dashboard disconnection."""
            print("[DataEmitter] Dashboard disconnected")
            self._session_manager.on_dashboard_disconnect()

    
    def submit_frame(self, frame_base64: str) -> None:
        """
        Submit a frame for emission (thread-safe).
        Call this from the main thread after capturing/encoding.
        
        Args:
            frame_base64: Base64 encoded JPEG frame
        """
        with self._frame_lock:
            self._pending_frame = frame_base64
            self._pending_frame_timestamp = time.time()
    
    def start(self) -> bool:
        """
        Start data emission loop.
        
        Returns:
            True if started successfully
        """
        with self._loop_lock:
            if self._running:
                print("[DataEmitter] Already running")
                return False
            
            print(f"[DataEmitter] Connecting to {self._server_url}...")
            
            if not self._socketio_client.connect(timeout=10.0):
                print("[DataEmitter] Failed to connect to server")
                return False
            
            print("[DataEmitter] Connected successfully")
            
            self._running = True
            self._loop_thread = threading.Thread(
                target=self._emission_loop,
                daemon=True
            )
            self._loop_thread.start()
            
            print(f"[DataEmitter] Started emission loop (metrics: {self._target_hz} Hz)")
            print("[DataEmitter] Frame streaming: Use bridge.submit_frame() from main thread")
            print("[DataEmitter] Episode tracking: Using episode_length_buf transition detection")
            
            return True
    
    def stop(self) -> None:
        """Stop data emission loop."""
        with self._loop_lock:
            if not self._running:
                return
            
            print("[DataEmitter] Stopping...")
            self._running = False
        
        if self._loop_thread:
            self._loop_thread.join(timeout=5.0)
        
        self._socketio_client.disconnect()
        
        print(f"[DataEmitter] Stopped")
        print(f"  Total metrics emissions: {self._emission_count}")
        print(f"  Total frame emissions: {self._frame_emission_count}")
        print(f"  Total errors: {self._error_count}")
    
    def _emission_loop(self) -> None:
        """Main emission loop thread."""
        print("[DataEmitter] Emission loop started")
        
        while self._running:
            loop_start_time = time.time()
            
            try:
                self._emit_metrics()
                self._emit_pending_frame()
                
            except Exception as e:
                self._error_count += 1
                print(f"[DataEmitter] Error in emission loop: {e}")
            
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, self._target_period - elapsed)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("[DataEmitter] Emission loop stopped")
    
    def _emit_pending_frame(self) -> None:
        """Emit any pending frame from the queue."""
        if not self._socketio_client.is_connected():
            return
        
        # Get pending frame (thread-safe)
        frame_base64 = None
        timestamp = 0.0
        
        with self._frame_lock:
            if self._pending_frame:
                frame_base64 = self._pending_frame
                timestamp = self._pending_frame_timestamp
                self._pending_frame = None  # Clear after taking
        
        if frame_base64:
            frame_data = {
                'type': 'frame_update',
                'timestamp': timestamp,
                'session_id': self._session_manager.get_session_id(),
                'env_index': self._env_index,
                'frame': frame_base64,
                'format': 'jpeg',
                'encoding': 'base64'
            }
            
            success = self._socketio_client.emit('frame_update', frame_data)
            
            if success:
                self._frame_emission_count += 1
                
                # Periodic frame status logging
                if self._frame_emission_count % 100 == 0:
                    print(f"[DataEmitter] Frame #{self._frame_emission_count} emitted")
    
    def _capture_termination_state(self, env: Any) -> None:
        """
        Capture and latch termination signals.
        Call this every iteration to catch termination signals.
        """
        try:
            if not hasattr(env, 'termination_manager'):
                return
            
            term_manager = env.termination_manager
            
            # Method 1: Use get_term() if available
            if hasattr(term_manager, 'get_term'):
                for term_name in ['goal_reached', 'collision', 'time_out']:
                    try:
                        term_value = term_manager.get_term(term_name)
                        if term_value is not None:
                            if term_value[self._env_index].item():
                                self._accumulated_termination[term_name] = True
                    except:
                        pass
            
            # Method 2: Direct access to _term_dones
            if hasattr(term_manager, '_term_dones'):
                term_dones = term_manager._term_dones
                for term_name in ['goal_reached', 'collision', 'time_out']:
                    if term_name in term_dones:
                        try:
                            if term_dones[term_name][self._env_index].item():
                                self._accumulated_termination[term_name] = True
                        except:
                            pass
        except:
            pass
    
    def _clear_accumulated_termination(self) -> None:
        """Clear accumulated termination signals."""
        self._accumulated_termination = {
            'goal_reached': False,
            'collision': False,
            'time_out': False
        }
    
    def _get_termination_reason(self) -> str:
        """Determine episode termination reason from accumulated signals."""
        if self._accumulated_termination.get('goal_reached', False):
            return 'success'
        elif self._accumulated_termination.get('collision', False):
            return 'collision'
        elif self._accumulated_termination.get('time_out', False):
            return 'timeout'
        return 'unknown'
    
    def _detect_episode_end(self, env: Any) -> tuple:
        """
        Detect episode end using episode_length_buf transition.
        
        Returns:
            (episode_ended: bool, termination_reason: str, episode_length: int)
        """
        try:
            if not hasattr(env, 'episode_length_buf'):
                return (False, None, 0)
            
            current_episode_length = env.episode_length_buf[self._env_index].item()
            
            # Initialize tracking on first call
            if not self._tracking_initialized:
                self._prev_episode_length = current_episode_length
                self._tracking_initialized = True
                return (False, None, 0)
            
            # Detect episode end: significant drop in episode length
            episode_ended = False
            termination_reason = None
            ended_episode_length = 0
            
            # Episode ended if current length is much smaller than previous
            if current_episode_length < self._prev_episode_length - self._reset_threshold:
                episode_ended = True
                ended_episode_length = self._prev_episode_length
                termination_reason = self._get_termination_reason()
            
            # Update tracking
            self._prev_episode_length = current_episode_length
            
            return (episode_ended, termination_reason, ended_episode_length)
            
        except Exception as e:
            print(f"[DataEmitter] Episode detection error: {e}")
            return (False, None, 0)
    
    def _emit_metrics(self) -> None:
        """Extract and emit metrics with proper episode tracking."""
        if not self._socketio_client.is_connected():
            return
        
        env = self._env_accessor.get_environment()
        if env is None:
            return
        
        try:
            # Capture termination signals every iteration (latching)
            self._capture_termination_state(env)
            
            # Detect episode end via episode_length_buf transition
            episode_ended, termination_reason, episode_length = self._detect_episode_end(env)
            
            if episode_ended:
                # Update counters based on accumulated termination reason
                if termination_reason == 'success':
                    self._session_manager.increment_counter('success')
                    print(f"[DataEmitter] ✓ SUCCESS! Total: {self._session_manager.get_counter('success')}")
                elif termination_reason == 'collision':
                    self._session_manager.increment_counter('collision')
                    print(f"[DataEmitter] ✗ COLLISION! Total: {self._session_manager.get_counter('collision')}")
                elif termination_reason == 'timeout':
                    self._session_manager.increment_counter('timeout')
                    print(f"[DataEmitter] ⏱ TIMEOUT! Total: {self._session_manager.get_counter('timeout')}")
                
                self._session_manager.increment_counter('total_episodes')
                print(f"[DataEmitter] Total episodes: {self._session_manager.get_counter('total_episodes')}")
                
                # Update episode history for mean calculation
                step_dt = self._metrics_extractor.get_step_dt(env)
                episode_time = episode_length * step_dt
                self._metrics_extractor.update_episode_history(episode_time=episode_time)
                
                # Clear accumulated state for next episode
                self._clear_accumulated_termination()
            
            # Extract and emit metrics
            metrics_data = self._extract_all_metrics(env)
            
            if metrics_data is None:
                return
            
            success = self._socketio_client.emit('metrics_update', metrics_data)
            
            if success:
                self._emission_count += 1
                self._last_emission_time = time.time()
                
                # Periodic status logging
                if self._emission_count % 300 == 0:
                    print(f"[DataEmitter] Status @ {self._emission_count} emissions:")
                    print(f"  Episodes: {self._session_manager.get_counter('total_episodes')}")
                    print(f"  Success: {self._session_manager.get_counter('success')}")
                    print(f"  Collision: {self._session_manager.get_counter('collision')}")
                    print(f"  Timeout: {self._session_manager.get_counter('timeout')}")
                    print(f"  Frames sent: {self._frame_emission_count}")
            
        except Exception as e:
            self._error_count += 1
            print(f"[DataEmitter] Emission error: {e}")
    
    def _debug_robot_access(self, env: Any) -> None:
        """Debug helper to understand robot access pattern."""
        if self._debug_logged_once:
            return
        
        print("\n[DataEmitter] === DEBUG: Robot Access ===")
        
        try:
            if hasattr(env, 'scene'):
                scene = env.scene
                print(f"  env.scene type: {type(scene)}")
                print(f"  env.scene dir: {[a for a in dir(scene) if not a.startswith('_')][:20]}")
                
                try:
                    robot = scene["robot"]
                    print(f"  scene['robot'] SUCCESS: {type(robot)}")
                    if hasattr(robot, 'data'):
                        data = robot.data
                        print(f"    robot.data type: {type(data)}")
                        print(f"    robot.data has root_lin_vel_b: {hasattr(data, 'root_lin_vel_b')}")
                        print(f"    robot.data has root_pos_w: {hasattr(data, 'root_pos_w')}")
                        if hasattr(data, 'root_lin_vel_b'):
                            vel = data.root_lin_vel_b
                            print(f"    root_lin_vel_b shape: {vel.shape}")
                            print(f"    root_lin_vel_b[0]: {vel[0]}")
                        if hasattr(data, 'root_pos_w'):
                            pos = data.root_pos_w
                            print(f"    root_pos_w shape: {pos.shape}")
                            print(f"    root_pos_w[0]: {pos[0]}")
                except Exception as e:
                    print(f"  scene['robot'] FAILED: {e}")
                
                if hasattr(scene, 'robot'):
                    print(f"  scene.robot EXISTS: {type(scene.robot)}")
                else:
                    print(f"  scene.robot NOT FOUND")
                
                if hasattr(scene, 'articulations'):
                    print(f"  scene.articulations: {list(scene.articulations.keys()) if hasattr(scene.articulations, 'keys') else scene.articulations}")
            else:
                print(f"  env has no 'scene' attribute")
                print(f"  env type: {type(env)}")
                print(f"  env dir: {[a for a in dir(env) if not a.startswith('_')][:20]}")
        except Exception as e:
            print(f"  Debug error: {e}")
        
        print("[DataEmitter] === END DEBUG ===\n")
        self._debug_logged_once = True
    
    def _extract_all_metrics(self, env: Any) -> Optional[Dict[str, Any]]:
        """Extract all metrics from environment."""
        try:
            if self._debug_extraction and not self._debug_logged_once:
                self._debug_robot_access(env)
            
            success_count = self._session_manager.get_counter('success')
            collision_count = self._session_manager.get_counter('collision')
            
            episode_time = self._metrics_extractor.extract_episode_time(
                env, self._env_index
            )
            
            mean_episode_time = self._metrics_extractor.extract_mean_episode_time()
            
            velocity = self._metrics_extractor.extract_robot_velocity(
                env, self._env_index
            )
            
            position = self._metrics_extractor.extract_robot_position(
                env, self._env_index
            )
            
            if self._debug_extraction and self._emission_count == 0:
                print(f"[DataEmitter] First extraction - velocity: {velocity}, position: {position}")
            
            data = {
                'type': 'metrics_update',
                'timestamp': time.time(),
                'session_id': self._session_manager.get_session_id(),
                'env_index': self._env_index,
                'metrics': {
                    'success_count': success_count,
                    'collision_count': collision_count,
                    'episode_time': episode_time,
                    'mean_episode_time': mean_episode_time,
                    'velocity': velocity,
                    'position': position
                },
                'session_stats': self._session_manager.get_statistics()
            }
            
            return data
            
        except Exception as e:
            print(f"[DataEmitter] Extraction error: {e}")
            return None
    
    def is_running(self) -> bool:
        """Check if emitter is running."""
        with self._loop_lock:
            return self._running
    
    def get_stats(self) -> Dict[str, Any]:
        """Get emitter statistics."""
        return {
            'running': self.is_running(),
            'connected': self._socketio_client.is_connected(),
            'session_id': self._session_manager.get_session_id(),
            'emission_count': self._emission_count,
            'frame_emission_count': self._frame_emission_count,
            'error_count': self._error_count,
            'target_hz': self._target_hz,
            'env_index': self._env_index,
            'last_emission_time': self._last_emission_time,
            'connection_status': self._socketio_client.get_status().value
        }
    