# isaaclab_bridge/frame_capture.py
"""
Viewport frame capture for Isaac Lab.
Encodes frames for streaming - frames must be captured from main thread.
"""

import base64
import io
import time
import numpy as np
from typing import Optional, Tuple, Any
from PIL import Image


class FrameCapture:
    """
    Encodes viewport frames for streaming.
    
    IMPORTANT: Frame capture (env.render()) must be called from the MAIN THREAD
    in Isaac Sim. This class handles encoding only. The actual capture should
    happen in the main simulation loop.
    """
    
    def __init__(
        self,
        target_width: int = 1280,
        target_height: int = 720,
        jpeg_quality: int = 75,
        capture_interval: float = 1.0 / 15.0  # 15 FPS default
    ):
        """
        Initialize frame capture/encoder.
        
        Args:
            target_width: Target frame width for streaming
            target_height: Target frame height for streaming
            jpeg_quality: JPEG compression quality (1-100)
            capture_interval: Minimum interval between captures in seconds
        """
        self._target_width = target_width
        self._target_height = target_height
        self._jpeg_quality = jpeg_quality
        self._capture_interval = capture_interval
        
        self._last_capture_time = 0.0
        self._frame_count = 0
        self._error_count = 0
        
        # Store the latest encoded frame for the emitter thread to send
        self._latest_frame: Optional[str] = None
        self._latest_frame_timestamp: float = 0.0
        
    def should_capture(self) -> bool:
        """
        Check if enough time has passed for next capture.
        Call this from the main thread before capturing.
        
        Returns:
            True if a new frame should be captured
        """
        current_time = time.time()
        if current_time - self._last_capture_time >= self._capture_interval:
            return True
        return False
    
    def encode_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Encode a captured frame as base64 JPEG.
        Call this from the main thread after env.render().
        
        Args:
            frame: RGB numpy array (H, W, 3) or (H, W, 4) for RGBA
            
        Returns:
            Base64 encoded JPEG string or None on error
        """
        try:
            if frame is None or frame.size == 0:
                return None
            
            # Handle different input formats
            if len(frame.shape) == 4:  # (1, H, W, C) or (N, H, W, C)
                frame = frame[0]
            
            if len(frame.shape) != 3:
                return None
            
            # Handle RGBA -> RGB
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            
            # Convert to uint8 if needed
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            elif frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Create PIL image
            img = Image.fromarray(frame)
            
            # Resize to target dimensions if needed
            if img.size != (self._target_width, self._target_height):
                img = img.resize(
                    (self._target_width, self._target_height),
                    Image.Resampling.LANCZOS
                )
            
            # Encode as JPEG
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=self._jpeg_quality)
            buffer.seek(0)
            
            # Base64 encode
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Update state
            self._frame_count += 1
            self._last_capture_time = time.time()
            self._latest_frame = encoded
            self._latest_frame_timestamp = self._last_capture_time
            
            return encoded
            
        except Exception as e:
            self._error_count += 1
            print(f"[FrameCapture] Encoding error: {e}")
            return None
    
    def capture_and_encode(self, env: Any) -> Optional[str]:
        """
        Capture frame from environment and encode it.
        MUST be called from the MAIN THREAD.
        
        Args:
            env: Isaac Lab environment with render() method
            
        Returns:
            Base64 encoded JPEG string or None
        """
        if not self.should_capture():
            return None
        
        try:
            # This MUST run on main thread
            frame = env.render()
            
            if frame is not None:
                return self.encode_frame(frame)
            
        except Exception as e:
            self._error_count += 1
            print(f"[FrameCapture] Capture error: {e}")
        
        return None
    
    def get_latest_frame(self) -> Tuple[Optional[str], float]:
        """
        Get the latest encoded frame (thread-safe read).
        Can be called from any thread.
        
        Returns:
            Tuple of (base64_frame, timestamp) or (None, 0)
        """
        return self._latest_frame, self._latest_frame_timestamp
    
    def clear_latest_frame(self) -> None:
        """Clear the latest frame after it's been sent."""
        self._latest_frame = None
    
    def get_stats(self) -> dict:
        """Get capture statistics."""
        return {
            'frame_count': self._frame_count,
            'error_count': self._error_count,
            'target_resolution': f"{self._target_width}x{self._target_height}",
            'target_fps': 1.0 / self._capture_interval if self._capture_interval > 0 else 0,
            'jpeg_quality': self._jpeg_quality
        }
    
    def set_resolution(self, width: int, height: int) -> None:
        """Update target resolution."""
        self._target_width = width
        self._target_height = height
    
    def set_fps(self, fps: float) -> None:
        """Update target FPS."""
        if fps > 0:
            self._capture_interval = 1.0 / fps
    
    def set_quality(self, quality: int) -> None:
        """Update JPEG quality (1-100)."""
        self._jpeg_quality = max(1, min(100, quality))