# behavior_logger.py
"""
Standalone behavior logger for YHBot model analysis.
Extracts comprehensive metrics per timestep and saves to CSV.

TEMPORARY FILE - Delete after analysis is complete.

Usage:
    Copy to: ~/Documents/fyp/yhbot_navigation/scripts/analysis/
    Run with: python play_with_logging.py --task Isaac-Navigation-Grid-YHBot-v0 --num_envs 1
"""

import csv
import os
import time
import math
from datetime import datetime
from typing import Optional, Dict, Any, List, Deque
from collections import deque
from dataclasses import dataclass, field, asdict
import torch


@dataclass
class TimestepData:
    """Data collected at each timestep."""
    # Episode & Time
    episode_number: int = 0
    timestep: int = 0
    time_elapsed: float = 0.0
    
    # Robot Velocity (robot frame)
    linear_velocity_x: float = 0.0
    linear_velocity_y: float = 0.0
    linear_speed: float = 0.0
    angular_velocity_z: float = 0.0
    
    # Velocity Changes
    velocity_change: float = 0.0
    acceleration: float = 0.0
    
    # Position & Navigation (world frame)
    robot_pos_x: float = 0.0
    robot_pos_y: float = 0.0
    goal_pos_x: float = 0.0
    goal_pos_y: float = 0.0
    distance_to_goal: float = 0.0
    
    # Progress Metrics (time windows)
    distance_change_1s: float = 0.0
    distance_change_2s: float = 0.0
    net_displacement_1s: float = 0.0
    net_displacement_2s: float = 0.0
    
    # Actions
    action_left_wheel: float = 0.0
    action_right_wheel: float = 0.0
    action_magnitude: float = 0.0
    
    # Sensor Data
    lidar_min_distance: float = 0.0
    lidar_front_min: float = 0.0
    contact_force_chassis: float = 0.0
    contact_force_roof: float = 0.0
    contact_detected: int = 0
    
    # Behavioral Flags
    is_near_goal: int = 0
    is_turning: int = 0
    is_moving_forward: int = 0
    is_moving_backward: int = 0
    is_slow: int = 0
    is_trying_to_move: int = 0
    
    # Episode Events
    episode_terminated: int = 0
    termination_reason: str = "none"


class BehaviorLogger:
    """
    Comprehensive behavior logger for model analysis.
    
    Extracts all metrics per timestep and saves to CSV file.
    """
    
    def __init__(
        self,
        output_dir: str = "analysis_logs",
        env_index: int = 0,
        step_dt: float = 1.0 / 15.0,  # Default ~0.0667s
        max_history_steps: int = 60,   # 4 seconds of history at 15Hz
    ):
        """
        Initialize behavior logger.
        
        Args:
            output_dir: Directory for CSV output
            env_index: Environment index to monitor
            step_dt: Time per step in seconds
            max_history_steps: Maximum history buffer size
        """
        self._output_dir = output_dir
        self._env_index = env_index
        self._step_dt = step_dt
        self._max_history = max_history_steps
        
        # Calculate steps for time windows
        self._steps_1s = int(1.0 / step_dt)  # ~15 steps
        self._steps_2s = int(2.0 / step_dt)  # ~30 steps
        
        # Episode tracking
        self._current_episode = 0
        self._current_timestep = 0
        self._episode_start_time = 0.0
        
        # History buffers for time-window calculations
        self._position_history: Deque[tuple] = deque(maxlen=max_history_steps)
        self._distance_history: Deque[float] = deque(maxlen=max_history_steps)
        self._speed_history: Deque[float] = deque(maxlen=max_history_steps)
        
        # Previous values for derivatives
        self._prev_speed = 0.0
        self._prev_episode_length = 0

        # Buffered row for delayed writing (to capture termination correctly)
        self._buffered_data: Optional[TimestepData] = None
        self._accumulated_termination: str = "none"
        
        # CSV file handling
        self._csv_file = None
        self._csv_writer = None
        self._filepath = ""
        self._row_count = 0
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open CSV file
        self._open_csv_file()
        
        print(f"[BehaviorLogger] Initialized")
        print(f"  Output dir: {output_dir}")
        print(f"  Step dt: {step_dt:.4f}s")
        print(f"  1s window: {self._steps_1s} steps")
        print(f"  2s window: {self._steps_2s} steps")
    
    def _open_csv_file(self) -> None:
        """Open CSV file with timestamp in filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yhbot_behavior_log_{timestamp}.csv"
        self._filepath = os.path.join(self._output_dir, filename)
        
        self._csv_file = open(self._filepath, 'w', newline='')
        self._csv_writer = csv.writer(self._csv_file)
        
        # Write header
        header = list(TimestepData().__dict__.keys())
        self._csv_writer.writerow(header)
        self._csv_file.flush()
        
        print(f"[BehaviorLogger] CSV file: {self._filepath}")
    
    def _get_robot(self, env: Any) -> Any:
        """Get robot articulation from environment."""
        try:
            if hasattr(env, 'scene'):
                return env.scene["robot"]
        except:
            pass
        return None
    
    def _extract_velocity(self, env: Any) -> tuple:
        """
        Extract robot velocity in robot frame.
        
        Returns:
            (vx, vy, speed, angular_z)
        """
        try:
            robot = self._get_robot(env)
            if robot is None:
                return (0.0, 0.0, 0.0, 0.0)
            
            # Linear velocity in robot frame (body frame)
            lin_vel = robot.data.root_lin_vel_b[self._env_index]
            vx = float(lin_vel[0].item())
            vy = float(lin_vel[1].item())
            speed = math.sqrt(vx**2 + vy**2)
            
            # Angular velocity in robot frame
            ang_vel = robot.data.root_ang_vel_b[self._env_index]
            angular_z = float(ang_vel[2].item())
            
            return (vx, vy, speed, angular_z)
            
        except Exception as e:
            return (0.0, 0.0, 0.0, 0.0)
    
    def _extract_position(self, env: Any) -> tuple:
        """
        Extract robot position in world frame.
        
        Returns:
            (pos_x, pos_y)
        """
        try:
            robot = self._get_robot(env)
            if robot is None:
                return (0.0, 0.0)
            
            pos = robot.data.root_pos_w[self._env_index]
            return (float(pos[0].item()), float(pos[1].item()))
            
        except Exception as e:
            return (0.0, 0.0)
    
    def _extract_goal(self, env: Any) -> tuple:
        """
        Extract goal position from command manager.
        
        Returns:
            (goal_x, goal_y)
        """
        try:
            if hasattr(env, 'command_manager'):
                command = env.command_manager.get_command("goal_pose")
                goal_x = float(command[self._env_index, 0].item())
                goal_y = float(command[self._env_index, 1].item())
                return (goal_x, goal_y)
        except:
            pass
        return (0.0, 0.0)
    
    def _extract_lidar(self, env: Any) -> tuple:
        """
        Extract LiDAR sensor data.
        
        Returns:
            (min_distance, front_min_distance)
        """
        try:
            # Try to get lidar from scene
            if hasattr(env, 'scene') and "lidar" in env.scene.sensors:
                lidar = env.scene.sensors["lidar"]
                if hasattr(lidar, 'data') and hasattr(lidar.data, 'ray_hits_w'):
                    # Get distances from ray caster
                    distances = lidar.data.ray_hits_w[self._env_index]
                    
                    # Calculate distances from robot position
                    robot_pos = self._get_robot(env).data.root_pos_w[self._env_index]
                    ray_distances = torch.norm(distances - robot_pos, dim=-1)
                    
                    # Filter out invalid (very large) values
                    valid_mask = ray_distances < 100.0
                    if valid_mask.any():
                        valid_distances = ray_distances[valid_mask]
                        min_dist = float(valid_distances.min().item())
                        
                        # Front arc (assuming 60 rays, front is middle ~15 rays)
                        num_rays = ray_distances.shape[0]
                        front_start = num_rays // 4
                        front_end = 3 * num_rays // 4
                        front_distances = ray_distances[front_start:front_end]
                        front_valid = front_distances[front_distances < 100.0]
                        front_min = float(front_valid.min().item()) if len(front_valid) > 0 else 100.0
                        
                        return (min_dist, front_min)
            
            # Try observations if lidar not directly accessible
            if hasattr(env, 'observation_manager'):
                obs = env.observation_manager.compute_group("policy")
                if obs is not None:
                    # First 60 values are typically lidar
                    lidar_obs = obs[self._env_index, :60]
                    # These are normalized, need to denormalize
                    # Assuming max range of 10m
                    distances = lidar_obs * 10.0
                    valid_mask = distances > 0.01
                    if valid_mask.any():
                        min_dist = float(distances[valid_mask].min().item())
                        # Front arc
                        front_distances = distances[15:45]
                        front_valid = front_distances[front_distances > 0.01]
                        front_min = float(front_valid.min().item()) if len(front_valid) > 0 else 10.0
                        return (min_dist, front_min)
                        
        except Exception as e:
            pass
        
        return (10.0, 10.0)  # Default to max range
    
    def _extract_contact(self, env: Any) -> tuple:
        """
        Extract contact sensor data.
        
        Returns:
            (chassis_force, roof_force, contact_detected)
        """
        try:
            if hasattr(env, 'scene'):
                chassis_force = 0.0
                roof_force = 0.0
                
                # Try chassis contact sensor
                if "contact_sensor" in env.scene.sensors:
                    sensor = env.scene.sensors["contact_sensor"]
                    if hasattr(sensor, 'data') and hasattr(sensor.data, 'net_forces_w'):
                        forces = sensor.data.net_forces_w[self._env_index]
                        chassis_force = float(torch.norm(forces).item())
                
                # Try roof contact sensor (might be separate)
                if "contact_sensor_roof" in env.scene.sensors:
                    sensor = env.scene.sensors["contact_sensor_roof"]
                    if hasattr(sensor, 'data') and hasattr(sensor.data, 'net_forces_w'):
                        forces = sensor.data.net_forces_w[self._env_index]
                        roof_force = float(torch.norm(forces).item())
                
                contact_detected = 1 if (chassis_force > 1.0 or roof_force > 1.0) else 0
                return (chassis_force, roof_force, contact_detected)
                
        except Exception as e:
            pass
        
        return (0.0, 0.0, 0)
    
    def _extract_termination(self, env: Any) -> str:
        """
        Check termination signals from environment.
        Note: This is called every step to try to catch signals.
        """
        try:
            if hasattr(env, 'termination_manager'):
                term_manager = env.termination_manager
                
                if hasattr(term_manager, '_term_dones'):
                    term_dones = term_manager._term_dones
                    
                    if 'goal_reached' in term_dones:
                        if term_dones['goal_reached'][self._env_index].item():
                            self._accumulated_termination = "goal_reached"
                            return self._accumulated_termination
                    
                    if 'collision' in term_dones:
                        if term_dones['collision'][self._env_index].item():
                            self._accumulated_termination = "collision"
                            return self._accumulated_termination
                    
                    if 'time_out' in term_dones:
                        if term_dones['time_out'][self._env_index].item():
                            self._accumulated_termination = "timeout"
                            return self._accumulated_termination
        except:
            pass
        
        return self._accumulated_termination

    def _calculate_time_window_metrics(
        self,
        current_pos: tuple,
        current_distance: float
    ) -> tuple:
        """
        Calculate metrics over time windows.
        
        Returns:
            (dist_change_1s, dist_change_2s, displacement_1s, displacement_2s)
        """
        dist_change_1s = 0.0
        dist_change_2s = 0.0
        displacement_1s = 0.0
        displacement_2s = 0.0
        
        # Distance change (negative = getting closer)
        if len(self._distance_history) >= self._steps_1s:
            dist_change_1s = current_distance - self._distance_history[-self._steps_1s]
        
        if len(self._distance_history) >= self._steps_2s:
            dist_change_2s = current_distance - self._distance_history[-self._steps_2s]
        
        # Net displacement (path length)
        if len(self._position_history) >= 2:
            # Calculate cumulative path length over 1s
            positions = list(self._position_history)
            
            # 1 second window
            start_idx = max(0, len(positions) - self._steps_1s)
            for i in range(start_idx + 1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                displacement_1s += math.sqrt(dx**2 + dy**2)
            
            # 2 second window
            start_idx = max(0, len(positions) - self._steps_2s)
            displacement_2s = 0.0
            for i in range(start_idx + 1, len(positions)):
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                displacement_2s += math.sqrt(dx**2 + dy**2)
        
        return (dist_change_1s, dist_change_2s, displacement_1s, displacement_2s)
    
    def _detect_episode_reset(self, env: Any) -> bool:
        """Detect if episode has reset."""
        try:
            if hasattr(env, 'episode_length_buf'):
                current_length = env.episode_length_buf[self._env_index].item()
                
                # Episode reset if length dropped significantly
                if current_length < self._prev_episode_length - 5:
                    self._prev_episode_length = current_length
                    return True
                
                self._prev_episode_length = current_length
        except:
            pass
        return False
    
    def _reset_episode_state(self) -> None:
        """Reset state for new episode."""
        self._current_episode += 1
        self._current_timestep = 0
        self._episode_start_time = time.time()
        
        # Clear history buffers
        self._position_history.clear()
        self._distance_history.clear()
        self._speed_history.clear()
        
        self._prev_speed = 0.0
        
        # Clear accumulated termination for new episode
        self._accumulated_termination = "none"
        
        print(f"[BehaviorLogger] Episode {self._current_episode} started")
    
    def log_timestep(
        self,
        env: Any,
        actions: Optional[torch.Tensor] = None
    ) -> TimestepData:
        """
        Log all metrics for current timestep.
        Uses buffered writing to correctly capture termination.
        """
        # Check termination signals (latching)
        self._extract_termination(env)
        
        # Check for episode reset
        episode_reset = self._detect_episode_reset(env)
        
        # If episode reset, finalize the buffered row as terminal
        if episode_reset and self._buffered_data is not None:
            self._buffered_data.episode_terminated = 1
            
            # Determine termination reason from accumulated signals OR final state
            if self._accumulated_termination != "none":
                self._buffered_data.termination_reason = self._accumulated_termination
            else:
                # Infer from final state if signals weren't caught
                if self._buffered_data.distance_to_goal < 0.55:  # Slightly above 0.5 threshold
                    self._buffered_data.termination_reason = "goal_reached"
                elif self._buffered_data.contact_detected == 1:
                    self._buffered_data.termination_reason = "collision"
                elif self._buffered_data.timestep >= 740:  # ~50s at 15Hz = 750 steps
                    self._buffered_data.termination_reason = "timeout"
                else:
                    self._buffered_data.termination_reason = "unknown"
            
            self._write_row(self._buffered_data)
            self._buffered_data = None
            self._reset_episode_state()
        elif self._buffered_data is not None:
            # Write previous buffered row (not terminal)
            self._write_row(self._buffered_data)
        
        # Create new data for current timestep
        data = TimestepData()
        
        # Episode & Time
        data.episode_number = self._current_episode
        data.timestep = self._current_timestep
        data.time_elapsed = self._current_timestep * self._step_dt
        
        # Velocity
        vx, vy, speed, ang_z = self._extract_velocity(env)
        data.linear_velocity_x = vx
        data.linear_velocity_y = vy
        data.linear_speed = speed
        data.angular_velocity_z = ang_z
        
        # Velocity changes
        data.velocity_change = speed - self._prev_speed
        data.acceleration = data.velocity_change / self._step_dt if self._step_dt > 0 else 0.0
        
        # Position
        pos_x, pos_y = self._extract_position(env)
        data.robot_pos_x = pos_x
        data.robot_pos_y = pos_y
        
        # Goal
        goal_x, goal_y = self._extract_goal(env)
        data.goal_pos_x = goal_x
        data.goal_pos_y = goal_y
        
        # Distance to goal
        data.distance_to_goal = math.sqrt((goal_x - pos_x)**2 + (goal_y - pos_y)**2)
        
        # Time window metrics
        dist_1s, dist_2s, disp_1s, disp_2s = self._calculate_time_window_metrics(
            (pos_x, pos_y), data.distance_to_goal
        )
        data.distance_change_1s = dist_1s
        data.distance_change_2s = dist_2s
        data.net_displacement_1s = disp_1s
        data.net_displacement_2s = disp_2s
        
        # Actions
        if actions is not None:
            try:
                act = actions[self._env_index] if len(actions.shape) > 1 else actions
                if len(act) >= 2:
                    data.action_left_wheel = float(act[0].item())
                    data.action_right_wheel = float(act[1].item())
                    data.action_magnitude = math.sqrt(
                        data.action_left_wheel**2 + data.action_right_wheel**2
                    )
            except:
                pass
        
        # Sensors
        lidar_min, lidar_front = self._extract_lidar(env)
        data.lidar_min_distance = lidar_min
        data.lidar_front_min = lidar_front
        
        chassis_force, roof_force, contact = self._extract_contact(env)
        data.contact_force_chassis = chassis_force
        data.contact_force_roof = roof_force
        data.contact_detected = contact
        
        # Behavioral flags
        data.is_near_goal = 1 if data.distance_to_goal < 1.0 else 0
        data.is_turning = 1 if abs(ang_z) > 0.5 else 0
        data.is_moving_forward = 1 if vx > 0.1 else 0
        data.is_moving_backward = 1 if vx < -0.05 else 0
        data.is_slow = 1 if speed < 0.1 else 0
        data.is_trying_to_move = 1 if data.action_magnitude > 0.3 else 0
        
        # Termination fields default to none (will be set on episode end)
        data.episode_terminated = 0
        data.termination_reason = "none"
        
        # Update history buffers
        self._position_history.append((pos_x, pos_y))
        self._distance_history.append(data.distance_to_goal)
        self._speed_history.append(speed)
        self._prev_speed = speed
        
        # Buffer this row (will be written next step or on close)
        self._buffered_data = data
        
        # Increment timestep
        self._current_timestep += 1
        
        return data
    
    def _write_row(self, data: TimestepData) -> None:
        """Write data row to CSV."""
        if self._csv_writer is None:
            return
        
        row = [
            data.episode_number,
            data.timestep,
            f"{data.time_elapsed:.3f}",
            f"{data.linear_velocity_x:.4f}",
            f"{data.linear_velocity_y:.4f}",
            f"{data.linear_speed:.4f}",
            f"{data.angular_velocity_z:.4f}",
            f"{data.velocity_change:.4f}",
            f"{data.acceleration:.4f}",
            f"{data.robot_pos_x:.3f}",
            f"{data.robot_pos_y:.3f}",
            f"{data.goal_pos_x:.3f}",
            f"{data.goal_pos_y:.3f}",
            f"{data.distance_to_goal:.3f}",
            f"{data.distance_change_1s:.4f}",
            f"{data.distance_change_2s:.4f}",
            f"{data.net_displacement_1s:.4f}",
            f"{data.net_displacement_2s:.4f}",
            f"{data.action_left_wheel:.4f}",
            f"{data.action_right_wheel:.4f}",
            f"{data.action_magnitude:.4f}",
            f"{data.lidar_min_distance:.3f}",
            f"{data.lidar_front_min:.3f}",
            f"{data.contact_force_chassis:.3f}",
            f"{data.contact_force_roof:.3f}",
            data.contact_detected,
            data.is_near_goal,
            data.is_turning,
            data.is_moving_forward,
            data.is_moving_backward,
            data.is_slow,
            data.is_trying_to_move,
            data.episode_terminated,
            data.termination_reason
        ]
        
        self._csv_writer.writerow(row)
        self._row_count += 1
        
        # Flush periodically
        if self._row_count % 100 == 0:
            self._csv_file.flush()
    
    def close(self) -> str:
        """
        Close logger and return filepath.
        
        Returns:
            Path to the saved CSV file
        """
        if self._buffered_data is not None:
            # Mark as terminated with accumulated reason (or timeout if still none)
            self._buffered_data.episode_terminated = 1
            if self._accumulated_termination != "none":
                self._buffered_data.termination_reason = self._accumulated_termination
            else:
                self._buffered_data.termination_reason = "interrupted"
            self._write_row(self._buffered_data)
            self._buffered_data = None

        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        
        print(f"\n[BehaviorLogger] Closed")
        print(f"  Total rows: {self._row_count}")
        print(f"  Total episodes: {self._current_episode + 1}")
        print(f"  File: {self._filepath}")
        
        return self._filepath
    
    def get_stats(self) -> dict:
        """Get logging statistics."""
        return {
            'filepath': self._filepath,
            'row_count': self._row_count,
            'current_episode': self._current_episode,
            'current_timestep': self._current_timestep
        }