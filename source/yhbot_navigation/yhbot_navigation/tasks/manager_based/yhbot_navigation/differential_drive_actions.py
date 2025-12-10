# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# ============================================================================
# DIFFERENTIAL DRIVE VELOCITY CONTROLLER VIA TORQUE
# ============================================================================
#
# This module provides custom action terms that implement velocity control
# by applying torques directly, bypassing the broken PhysX velocity drive.
#
# Control Law:
#   torque = Kp * (target_velocity - current_velocity)
#   torque = clamp(torque, -max_torque, max_torque)
#   robot.set_joint_effort_target(torque)
#
# ============================================================================

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DifferentialDriveVelocityAction(ActionTerm):
    """
    Velocity control for differential drive robots via torque application.
    
    This action term:
    1. Takes velocity commands from the policy (actions in [-1, 1])
    2. Scales them to target wheel velocities
    3. Reads current wheel velocities from the robot
    4. Computes torque using a P controller: τ = Kp * (target - current)
    5. Applies the torque directly to the wheels
    
    This bypasses the PhysX velocity drive which has issues with URDF imports.
    """
    
    cfg: DifferentialDriveVelocityActionCfg
    
    def __init__(self, cfg: DifferentialDriveVelocityActionCfg, env: ManagerBasedEnv):
        """Initialize the action term."""
        super().__init__(cfg, env)
        
        # Get robot asset
        self._asset = env.scene[cfg.asset_name]
        
        # Find joint indices
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        
        if len(self._joint_ids) != 2:
            raise ValueError(
                f"Expected 2 joints for differential drive, got {len(self._joint_ids)}: {self._joint_names}"
            )
        
        # Store parameters
        self._Kp = cfg.velocity_gain
        self._max_torque = cfg.max_torque
        self._vel_scale = cfg.velocity_scale
        
        # Action buffers
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._torques = torch.zeros(self.num_envs, 2, device=self.device)
        
        print(f"[DifferentialDriveVelocityAction] Initialized")
        print(f"  Joints: {self._joint_names}")
        print(f"  Kp: {self._Kp}, Max Torque: {self._max_torque} Nm, Vel Scale: {self._vel_scale} rad/s")
    
    @property
    def action_dim(self) -> int:
        return 2
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._torques
    
    def process_actions(self, actions: torch.Tensor):
        """Convert velocity commands to torques."""
        self._raw_actions[:] = actions
        
        # Scale actions to target velocities
        target_vel = actions * self._vel_scale
        
        # Get current velocities
        current_vel = self._asset.data.joint_vel[:, self._joint_ids]
        
        # P controller
        vel_error = target_vel - current_vel
        torques = self._Kp * vel_error
        
        # Clamp torques
        self._torques[:] = torch.clamp(torques, -self._max_torque, self._max_torque)
    
    def apply_actions(self):
        """Apply computed torques to the robot."""
        self._asset.set_joint_effort_target(self._torques, joint_ids=self._joint_ids)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset action buffers."""
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._torques[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._torques[env_ids] = 0.0


@configclass
class DifferentialDriveVelocityActionCfg(ActionTermCfg):
    """Configuration for differential drive velocity action."""
    
    class_type: type = DifferentialDriveVelocityAction
    
    asset_name: str = MISSING
    """Name of the robot asset in the scene."""
    
    joint_names: list[str] = MISSING
    """Names of wheel joints: [left_wheel_joint, right_wheel_joint]."""
    
    velocity_gain: float = 10.0
    """P controller gain. Torque = Kp * velocity_error. Range: 1-100."""
    
    max_torque: float = 20.0
    """Maximum torque per wheel in Nm."""
    
    velocity_scale: float = 5.0
    """Scale factor: actions [-1,1] map to velocities [-scale, scale] rad/s."""


# ============================================================================
# ALTERNATIVE: Direct Torque Control
# ============================================================================

class DirectTorqueAction(ActionTerm):
    """
    Direct torque control - policy outputs torques directly.
    
    Actions in [-1, 1] are scaled to [-max_torque, max_torque] Nm.
    """
    
    cfg: DirectTorqueActionCfg
    
    def __init__(self, cfg: DirectTorqueActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._max_torque = cfg.max_torque
        
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self._torques = torch.zeros(self.num_envs, 2, device=self.device)
        
        print(f"[DirectTorqueAction] Initialized - Joints: {self._joint_names}")
    
    @property
    def action_dim(self) -> int:
        return 2
    
    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._torques
    
    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._torques[:] = actions * self._max_torque
    
    def apply_actions(self):
        self._asset.set_joint_effort_target(self._torques, joint_ids=self._joint_ids)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._torques[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._torques[env_ids] = 0.0


@configclass
class DirectTorqueActionCfg(ActionTermCfg):
    """Configuration for direct torque control."""
    
    class_type: type = DirectTorqueAction
    
    asset_name: str = MISSING
    """Name of the robot asset."""
    
    joint_names: list[str] = MISSING
    """Names of wheel joints."""
    
    max_torque: float = 20.0
    """Max torque in Nm. Actions [-1,1] map to [-max, max]."""
