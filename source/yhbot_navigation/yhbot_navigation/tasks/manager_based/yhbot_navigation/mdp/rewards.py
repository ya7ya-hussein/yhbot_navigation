# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class progress_to_goal(ManagerTermBase):
    """
    Reward for reducing distance to goal (progress reward).
    
    This is a STATEFUL reward that tracks previous distance using the correct
    Isaac Lab pattern: inheriting from ManagerTermBase.
    
    Returns:
    - Positive value when robot gets closer to goal
    - Negative value when robot moves away from goal
    - Zero on first step of episode or after reset
    
    Official Pattern Reference:
    - isaaclab/envs/mdp/rewards.py :: is_terminated_term
    - isaaclab/envs/mdp/curriculums.py :: modify_reward_weight
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        """Initialize the progress reward term.
        
        Args:
            cfg: The reward term configuration.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        
        # Get command name from params
        self._command_name = cfg.params.get("command_name", "goal_pose")
        
        # Initialize previous distance buffer for all environments
        # This is the CORRECT way to store stateful data in Isaac Lab
        self._prev_distance = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the term for specific environments.
        
        This method is called automatically by Isaac Lab when environments reset.
        
        Args:
            env_ids: Environment indices to reset. If None, reset all.
        """
        if env_ids is None:
            self._prev_distance.zero_()
            self._initialized.zero_()
        else:
            self._prev_distance[env_ids] = 0.0
            self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "goal_pose",
    ) -> torch.Tensor:
        """Compute the progress reward.
        
        Args:
            env: The environment instance.
            command_name: Name of the command term for goal position.
            
        Returns:
            Tensor of shape (num_envs,) with distance reduction in meters.
        """
        # Get robot position (world frame, xy only)
        robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # [N, 2]
        
        # Get goal position from command
        command = env.command_manager.get_command(self._command_name)
        goal_pos = command[:, :2]  # [N, 2]
        
        # Calculate current distance
        current_distance = torch.norm(goal_pos - robot_pos, dim=-1)
        
        # Calculate progress (positive = getting closer)
        # For first step, return 0 (no previous distance to compare)
        progress = torch.where(
            self._initialized,
            self._prev_distance - current_distance,
            torch.zeros_like(current_distance)
        )
        
        # Update state for next step
        self._prev_distance = current_distance.clone()
        self._initialized[:] = True
        
        return progress


class potential_field_progress(ManagerTermBase):
    """
    Potential field-based progress reward.
    
    Uses attractive potential: U = 0.5 * mu * distance²
    Reward = U_prev - U_curr (decrease in potential = positive reward)
    
    This naturally scales reward with distance (larger steps when far, smaller when close).
    """

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        
        self._command_name = cfg.params.get("command_name", "goal_pose")
        self._mu = cfg.params.get("mu", 1.0)
        
        self._prev_potential = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._prev_potential.zero_()
            self._initialized.zero_()
        else:
            self._prev_potential[env_ids] = 0.0
            self._initialized[env_ids] = False

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str = "goal_pose",
        mu: float = 1.0,
    ) -> torch.Tensor:
        robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
        command = env.command_manager.get_command(self._command_name)
        goal_pos = command[:, :2]
        
        current_distance = torch.norm(goal_pos - robot_pos, dim=-1)
        current_potential = 0.5 * self._mu * current_distance ** 2
        
        # Reward is decrease in potential (closer = lower potential = positive reward)
        reward = torch.where(
            self._initialized,
            self._prev_potential - current_potential,
            torch.zeros_like(current_potential)
        )
        
        self._prev_potential = current_potential.clone()
        self._initialized[:] = True
        
        return reward

def goal_reached_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str = "goal_pose",
    threshold: float = 0.5,
) -> torch.Tensor:
    """Binary reward when robot reaches the goal.
    
    This is STATELESS - no previous values needed, so simple function is fine.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    distance = torch.norm(goal_pos - robot_pos, dim=-1)
    return (distance < threshold).float()


def heading_alignment_to_goal(
    env: "ManagerBasedRLEnv",
    command_name: str = "goal_pose",
) -> torch.Tensor:
    """Reward for facing toward the goal.
    
    This is STATELESS - computes from current state only.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    robot_quat = env.scene["robot"].data.root_quat_w
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    
    # Direction to goal
    to_goal = goal_pos - robot_pos
    to_goal_norm = torch.norm(to_goal, dim=-1, keepdim=True).clamp(min=1e-6)
    to_goal_dir = to_goal / to_goal_norm
    
    # Robot heading from quaternion
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    heading_dir = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    
    # Dot product = cos(angle)
    alignment = torch.sum(heading_dir * to_goal_dir, dim=-1)
    return alignment


# def collision_penalty(
#     env: "ManagerBasedRLEnv",
#     threshold: float = 1.0,
#     sensor_cfg = None,
# ) -> torch.Tensor:
#     """Penalty for collisions detected by contact sensor.
    
#     This is STATELESS - checks current contacts only.
#     """
#     from isaaclab.managers import SceneEntityCfg
    
#     contact_sensor = env.scene.sensors[sensor_cfg.name]
#     net_forces = contact_sensor.data.net_forces_w_history
#     force_magnitude = torch.norm(net_forces[:, 0, :, :], dim=-1)
#     max_force = torch.max(force_magnitude, dim=-1)[0]
#     return (max_force > threshold).float()


def distance_to_goal_penalty(
    env: "ManagerBasedRLEnv",
    command_name: str = "goal_pose",
) -> torch.Tensor:
    """Penalty based on current distance to goal.
    
    This is STATELESS - just returns current distance.
    NOTE: Consider using progress_to_goal (stateful) instead for better learning.
    """
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    command = env.command_manager.get_command(command_name)
    goal_pos = command[:, :2]
    return torch.norm(goal_pos - robot_pos, dim=-1)