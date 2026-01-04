# isaaclab_bridge/launch_bridge.py
"""
Launch script for Isaac Lab Bridge.
Run this in a separate terminal after starting Isaac Lab play script.
"""

import sys
import time
from bridge_main import IsaacLabBridge


def main():
    print("=" * 70)
    print("Isaac Lab Bridge - Launcher")
    print("=" * 70)
    print()
    
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3001"
    target_hz = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    env_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    
    print(f"Server URL: {server_url}")
    print(f"Target Hz: {target_hz}")
    print(f"Env Index: {env_index}")
    print()
    
    bridge = IsaacLabBridge(
        server_url=server_url,
        target_hz=target_hz,
        env_index=env_index,
        persist_data=True
    )
    
    print("Starting bridge...")
    print("Waiting for environment to be registered by Isaac Lab...")
    print()
    
    if bridge.start():
        print("✓ Bridge started successfully!")
        print()
        print("System ready:")
        print("  [1] Backend server running")
        print("  [2] Bridge connected and streaming")
        print("  [3] Waiting for Isaac Lab environment...")
        print()
        print("Now run Isaac Lab play script in another terminal")
        print("Press Ctrl+C to stop")
        print()
        
        try:
            last_episode_count = 0
            
            while True:
                time.sleep(5)
                status = bridge.get_status()
                
                if status['environment_registered']:
                    session = status.get('session', {})
                    counters = session.get('counters', {})
                    current_episodes = counters.get('total_episodes', 0)
                    
                    if current_episodes != last_episode_count:
                        print(f"\n[Session: {session.get('session_id')}]")
                        print(f"  Episodes: {current_episodes}")
                        print(f"  Success: {counters.get('success', 0)}")
                        print(f"  Collisions: {counters.get('collision', 0)}")
                        print(f"  Timeouts: {counters.get('timeout', 0)}")
                        last_episode_count = current_episodes
        
        except KeyboardInterrupt:
            print("\n\nShutting down bridge...")
            bridge.stop()
            print("Bridge stopped")
    
    else:
        print("✗ Failed to start bridge")
        print(f"Make sure backend server is running on {server_url}")
        sys.exit(1)


if __name__ == "__main__":
    main()