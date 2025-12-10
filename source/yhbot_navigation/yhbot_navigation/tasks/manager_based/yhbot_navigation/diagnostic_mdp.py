# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Diagnostic MDP functions for YHBot troubleshooting

"""Diagnostic observation and reward terms for debugging YHBot navigation."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def log_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Log and return robot base position and orientation.
    
    This diagnostic function prints detailed robot state every step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get root state (base_link position and orientation)
    root_pos = asset.data.root_pos_w  # (num_envs, 3) - world position
    root_quat = asset.data.root_quat_w  # (num_envs, 4) - world quaternion [w,x,y,z]
    root_lin_vel = asset.data.root_lin_vel_w  # (num_envs, 3) - linear velocity
    root_ang_vel = asset.data.root_ang_vel_w  # (num_envs, 3) - angular velocity
    
    # Print for first environment only (to avoid spam)
    if env.episode_length_buf[0] % 60 == 0:  # Print every 60 steps (1 second at 60Hz)
        print("\n" + "="*80)
        print(f"DIAGNOSTIC - Episode Step: {env.episode_length_buf[0].item()}")
        print("="*80)
        print(f"Robot Position (m):     X={root_pos[0,0]:+.4f}  Y={root_pos[0,1]:+.4f}  Z={root_pos[0,2]:+.4f}")
        print(f"Robot Velocity (m/s):   X={root_lin_vel[0,0]:+.4f}  Y={root_lin_vel[0,1]:+.4f}  Z={root_lin_vel[0,2]:+.4f}")
        print(f"Robot Ang Vel (rad/s):  X={root_ang_vel[0,0]:+.4f}  Y={root_ang_vel[0,1]:+.4f}  Z={root_ang_vel[0,2]:+.4f}")
        
        # Check if robot is falling/floating
        if root_pos[0,2] < 0.15:
            print(f"⚠️  WARNING: Robot too low! Z={root_pos[0,2]:.4f}m (sinking into ground?)")
        elif root_pos[0,2] > 0.35:
            print(f"⚠️  WARNING: Robot too high! Z={root_pos[0,2]:.4f}m (floating?)")
        
        # Check if robot is tilting
        # Convert quaternion to approximate tilt
        # For small angles: tilt_x ≈ 2*quat_x, tilt_y ≈ 2*quat_y
        tilt_x = 2.0 * root_quat[0,1]  # Roll
        tilt_y = 2.0 * root_quat[0,2]  # Pitch
        if abs(tilt_x) > 0.1 or abs(tilt_y) > 0.1:
            print(f"⚠️  WARNING: Robot tilting! Roll={tilt_x:.3f} rad, Pitch={tilt_y:.3f} rad")
    
    # Return Z position as observation (useful for monitoring)
    return root_pos[:, 2].unsqueeze(-1)


def log_wheel_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Log and return wheel joint states.
    
    Shows commanded vs actual wheel velocities and applied torques.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get wheel joint indices
    left_wheel_idx = asset.find_joints("middle_left_wheel_joint")[0][0]
    right_wheel_idx = asset.find_joints("middle_right_wheel_joint")[0][0]
    
    # Joint states
    joint_pos = asset.data.joint_pos  # (num_envs, num_joints)
    joint_vel = asset.data.joint_vel  # (num_envs, num_joints)
    applied_torque = asset.data.applied_torque  # (num_envs, num_joints)
    
    # Print for first environment
    if env.episode_length_buf[0] % 60 == 0:
        print("-"*80)
        print("WHEEL DIAGNOSTICS:")
        print(f"Left  Wheel: Pos={joint_pos[0,left_wheel_idx]:+.3f} rad  "
              f"Vel={joint_vel[0,left_wheel_idx]:+.3f} rad/s  "
              f"Torque={applied_torque[0,left_wheel_idx]:+.1f} Nm")
        print(f"Right Wheel: Pos={joint_pos[0,right_wheel_idx]:+.3f} rad  "
              f"Vel={joint_vel[0,right_wheel_idx]:+.3f} rad/s  "
              f"Torque={applied_torque[0,right_wheel_idx]:+.1f} Nm")
        
        # Calculate linear velocity from wheels (theoretical)
        wheel_radius = 0.1  # meters
        left_vel_linear = joint_vel[0,left_wheel_idx] * wheel_radius
        right_vel_linear = joint_vel[0,right_wheel_idx] * wheel_radius
        avg_vel = (left_vel_linear + right_vel_linear) / 2.0
        print(f"Theoretical Speed: {avg_vel:.4f} m/s")
        
        # Check if wheels are spinning
        if abs(joint_vel[0,left_wheel_idx]) < 0.01 and abs(joint_vel[0,right_wheel_idx]) < 0.01:
            print("⚠️  WARNING: Wheels not spinning! (Robot stuck or damping too high?)")
        
        # Check if torques are reasonable
        if abs(applied_torque[0,left_wheel_idx]) > 15000 or abs(applied_torque[0,right_wheel_idx]) > 15000:
            print("⚠️  WARNING: Very high torques! (Robot struggling?)")
    
    # Return wheel velocities as observation
    wheel_vels = torch.stack([
        joint_vel[:, left_wheel_idx],
        joint_vel[:, right_wheel_idx]
    ], dim=-1)
    return wheel_vels


def log_contact_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Log and return contact sensor data.
    
    Shows which parts of robot are in contact with ground.
    """
    contact_sensor = env.scene.sensors["contact_sensor"]
    
    # Get contact forces
    net_contact_forces = contact_sensor.data.net_forces_w  # (num_envs, num_bodies, 3)
    
    # Print for first environment
    if env.episode_length_buf[0] % 60 == 0:
        print("-"*80)
        print("CONTACT DIAGNOSTICS:")
        
        # Count how many bodies are in contact
        contact_threshold = 0.1  # Newtons
        contact_forces_magnitude = torch.norm(net_contact_forces[0], dim=-1)
        num_contacts = (contact_forces_magnitude > contact_threshold).sum().item()
        
        print(f"Number of bodies in contact: {num_contacts}")
        
        # Show contact forces for first few bodies
        for i in range(min(6, net_contact_forces.shape[1])):
            force_mag = contact_forces_magnitude[i].item()
            if force_mag > contact_threshold:
                fx, fy, fz = net_contact_forces[0,i,0], net_contact_forces[0,i,1], net_contact_forces[0,i,2]
                print(f"  Body {i}: Force=({fx:+6.1f}, {fy:+6.1f}, {fz:+6.1f}) N  Magnitude={force_mag:.1f} N")
        
        # Check if robot is bouncing (oscillating vertical forces)
        total_vertical_force = net_contact_forces[0, :, 2].sum().item()
        expected_weight = 75.0 * 9.81  # 75kg robot
        if abs(total_vertical_force - expected_weight) > expected_weight * 0.5:
            print(f"⚠️  WARNING: Abnormal vertical force! Got {total_vertical_force:.1f}N, expected ~{expected_weight:.1f}N")
    
    # Return total contact force magnitude as observation
    total_contact_force = torch.norm(net_contact_forces.sum(dim=1), dim=-1)
    return total_contact_force.unsqueeze(-1)


def log_action_commands(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Log commanded actions vs actual results.
    
    Shows what the policy commanded and what actually happened.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get the last action that was applied
    if hasattr(env, 'action_manager') and hasattr(env.action_manager, '_action'):
        last_action = env.action_manager._action
        
        # Print for first environment
        if env.episode_length_buf[0] % 60 == 0:
            print("-"*80)
            print("ACTION DIAGNOSTICS:")
            print(f"Commanded Action (normalized): Left={last_action[0,0]:+.4f}  Right={last_action[0,1]:+.4f}")
            
            # Calculate what velocity this should produce
            scale = 10.0  # from ActionsCfg
            cmd_left_vel = last_action[0,0] * scale
            cmd_right_vel = last_action[0,1] * scale
            print(f"Commanded Velocities (rad/s): Left={cmd_left_vel:+.2f}  Right={cmd_right_vel:+.2f}")
            
            # Get actual wheel velocities
            left_wheel_idx = asset.find_joints("middle_left_wheel_joint")[0][0]
            right_wheel_idx = asset.find_joints("middle_right_wheel_joint")[0][0]
            actual_left_vel = asset.data.joint_vel[0, left_wheel_idx]
            actual_right_vel = asset.data.joint_vel[0, right_wheel_idx]
            print(f"Actual Velocities (rad/s):    Left={actual_left_vel:+.2f}  Right={actual_right_vel:+.2f}")
            
            # Calculate tracking error
            left_error = abs(actual_left_vel - cmd_left_vel)
            right_error = abs(actual_right_vel - cmd_right_vel)
            if left_error > 1.0 or right_error > 1.0:
                print(f"⚠️  WARNING: Large velocity tracking error! Left={left_error:.2f}, Right={right_error:.2f}")
            
            print("="*80 + "\n")
    
    # Return zero (dummy observation)
    return torch.zeros(env.num_envs, 1, device=env.device)


def reward_stability_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalty for unstable motion (shaking, bouncing, tilting).
    
    Helps identify if robot is experiencing instability.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Penalize vertical velocity (bouncing)
    vertical_vel = asset.data.root_lin_vel_w[:, 2]
    bounce_penalty = torch.abs(vertical_vel) * 10.0
    
    # Penalize angular velocity around X and Y axes (tilting/rolling)
    tilt_vel = torch.abs(asset.data.root_ang_vel_w[:, 0]) + torch.abs(asset.data.root_ang_vel_w[:, 1])
    tilt_penalty = tilt_vel * 5.0
    
    total_penalty = bounce_penalty + tilt_penalty
    
    # Print warnings
    if env.episode_length_buf[0] % 60 == 0:
        if total_penalty[0] > 1.0:
            print(f"⚠️  STABILITY WARNING: Penalty={total_penalty[0]:.2f} "
                  f"(Bounce={bounce_penalty[0]:.2f}, Tilt={tilt_penalty[0]:.2f})")
    
    return -total_penalty
