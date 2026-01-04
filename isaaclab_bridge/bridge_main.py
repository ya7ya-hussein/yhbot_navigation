# isaaclab_bridge/bridge_main.py
"""
Main bridge entry point.
Combines all components for complete bridge functionality.
"""

import time
import sys
from typing import Optional

from env_registry import EnvironmentRegistry
from env_accessor import EnvironmentAccessor
from data_emitter import DataEmitter
from session_manager import SessionManager
from frame_capture import FrameCapture


class IsaacLabBridge:
    """
    Main Isaac Lab Bridge.
    Manages environment access, metrics extraction, and data emission.
    
    Frame capture must be done on the main thread. Use:
        bridge.capture_and_submit_frame(env)
    in your main simulation loop.
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:3001",
        target_hz: float = 30.0,
        env_index: int = 0,
        persist_data: bool = False,
        frame_width: int = 1280,
        frame_height: int = 720,
        frame_fps: float = 15.0,
        frame_quality: int = 75
    ):
        """
        Initialize bridge.
        
        Args:
            server_url: Backend server URL
            target_hz: Target emission frequency
            env_index: Environment index to monitor
            persist_data: Enable session data persistence
            frame_width: Target frame width for streaming
            frame_height: Target frame height for streaming
            frame_fps: Target FPS for frame capture
            frame_quality: JPEG quality (1-100)
        """
        self._server_url = server_url
        self._target_hz = target_hz
        self._env_index = env_index
        
        self._registry = EnvironmentRegistry()
        self._accessor = EnvironmentAccessor()
        self._session_manager = SessionManager(persist_data=persist_data)
        self._emitter: Optional[DataEmitter] = None
        
        # Frame capture (encoding only - capture happens on main thread)
        self._frame_capture = FrameCapture(
            target_width=frame_width,
            target_height=frame_height,
            jpeg_quality=frame_quality,
            capture_interval=1.0 / frame_fps
        )
        self._frame_capture_enabled = True
        

    
    def register_environment(self, env) -> None:
        """
        Register Isaac Lab environment.
        
        Args:
            env: Environment instance
        """
        self._registry.register(env)
        print(f"[Bridge] Environment registered")
        
        verification = self._accessor.verify_environment_access()
        
        if verification['can_access_env']:
            print("[Bridge] ✓ Environment access verified")
            props = verification['env_properties']
            for key, value in props.items():
                print(f"  {key}: {value}")
        else:
            print("[Bridge] ✗ Environment access verification failed")
            if 'access_error' in verification:
                print(f"  Error: {verification['access_error']}")
    
    def start(self) -> bool:
        """
        Start bridge data emission.
        
        Returns:
            True if started successfully
        """
        if self._emitter and self._emitter.is_running():
            print("[Bridge] Already running")
            return False
        
        print(f"[Bridge] Starting...")
        print(f"  Server: {self._server_url}")
        print(f"  Target Hz: {self._target_hz}")
        print(f"  Env Index: {self._env_index}")
        
        session_id = self._session_manager.start_session()
        
        self._emitter = DataEmitter(
            server_url=self._server_url,
            target_hz=self._target_hz,
            env_index=self._env_index,
            session_manager=self._session_manager
        )
        
        success = self._emitter.start()
        

        if success:
            print("[Bridge] ✓ Started successfully")
            if self._frame_capture_enabled:
                print("[Bridge] Frame capture: Call bridge.capture_and_submit_frame(env) in main loop")
        else:
            print("[Bridge] ✗ Failed to start")
            self._session_manager.stop_session()
        
        return success
    

    
    def stop(self) -> None:
        """Stop bridge data emission."""
        if self._emitter:
            self._emitter.stop()
            self._emitter = None
        
        self._session_manager.stop_session()
        
        # Print frame capture stats
        if self._frame_capture_enabled:
            stats = self._frame_capture.get_stats()
            print(f"[Bridge] Frame capture stats:")
            print(f"  Frames captured: {stats['frame_count']}")
            print(f"  Errors: {stats['error_count']}")
        
        print("[Bridge] Stopped")
    
    def capture_and_submit_frame(self, env) -> bool:
        """
        Capture a frame from the environment and submit it for emission.
        
        MUST be called from the MAIN THREAD (in the simulation loop).
        Respects the configured frame rate - will skip if called too frequently.
        
        Args:
            env: Isaac Lab environment with render() method
            
        Returns:
            True if a frame was captured and submitted
        """
        if not self._frame_capture_enabled:
            return False
        
        if not self._emitter or not self._emitter.is_running():
            return False
        
        # Check if we should capture (respects FPS limit)
        if not self._frame_capture.should_capture():
            return False
        
        try:
            # Capture frame (this calls env.render() on main thread)
            frame = env.render()
            
            if frame is not None:
                # Encode frame
                frame_base64 = self._frame_capture.encode_frame(frame)
                
                if frame_base64:
                    # Submit to emitter for transmission
                    self._emitter.submit_frame(frame_base64)
                    return True
        
        except Exception as e:
            print(f"[Bridge] Frame capture error: {e}")
        
        return False
    
    def submit_frame(self, frame_base64: str) -> bool:
        """
        Submit a pre-encoded frame for emission.
        
        Use this if you're encoding frames yourself.
        
        Args:
            frame_base64: Base64 encoded JPEG frame
            
        Returns:
            True if submitted successfully
        """
        if not self._emitter or not self._emitter.is_running():
            return False
        
        self._emitter.submit_frame(frame_base64)
        return True
    
    def set_frame_capture_enabled(self, enabled: bool) -> None:
        """Enable or disable frame capture."""
        self._frame_capture_enabled = enabled
        print(f"[Bridge] Frame capture {'enabled' if enabled else 'disabled'}")
    
    def reset_counters(self) -> None:
        """Reset session counters."""
        self._session_manager.reset_counters()
    
    def on_dashboard_connect(self) -> None:
        """Handle dashboard connection event."""
        self._session_manager.on_dashboard_connect()
    
    def on_dashboard_disconnect(self) -> None:
        """Handle dashboard disconnection event."""
        self._session_manager.on_dashboard_disconnect()
    
    def on_environment_restart(self) -> None:
        """Handle environment restart event."""
        self._session_manager.on_environment_restart()
    
    def get_status(self) -> dict:
        """Get bridge status."""
        status = {
            'environment_registered': self._registry.is_registered(),
            'running': False,
            'session': self._session_manager.get_statistics(),
            'emitter_stats': {},
            'frame_capture': self._frame_capture.get_stats()
        }
        
        if self._emitter:
            status['running'] = self._emitter.is_running()
            status['emitter_stats'] = self._emitter.get_stats()
        
        return status

def main():
    """Main entry point for standalone testing."""
    print("=" * 60)
    print("Isaac Lab Bridge - Standalone Test Mode")
    print("=" * 60)
    
    bridge = IsaacLabBridge(
        server_url="http://localhost:3001",
        target_hz=30.0,
        env_index=0,
        persist_data=True
    )
    
    print("\nNote: No environment registered (test mode)")
    print("Bridge will run but emit null metrics")
    
    if bridge.start():
        print("\nBridge running... Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(5)
                
                status = bridge.get_status()
                session_stats = status.get('session', {})
                
                print(f"\nSession Stats:")
                print(f"  ID: {session_stats.get('session_id')}")
                print(f"  Duration: {session_stats.get('session_duration', 0):.1f}s")
                print(f"  Episodes: {session_stats.get('counters', {}).get('total_episodes', 0)}")
                print(f"  Success: {session_stats.get('counters', {}).get('success', 0)}")
        
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            bridge.stop()
    
    else:
        print("\nFailed to start bridge")
        print("Make sure backend server is running on http://localhost:3001")


if __name__ == "__main__":
    main()