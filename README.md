# YHBot Navigation

Autonomous mobile robot navigation using Deep Reinforcement Learning with Isaac Lab.

## Overview

This project trains a differential drive robot to perform point-to-point navigation with dynamic obstacle avoidance using Proximal Policy Optimization (PPO). The robot learns to navigate across six progressively complex environments, from simple grids to realistic warehouses and outdoor spaces.

## Key Features

- **Deep RL Navigation**: PPO-based learning for adaptive obstacle avoidance
- **Progressive Training**: Curriculum learning across 6 diverse environments
- **Dynamic Obstacles**: Handles moving objects (humans, forklifts, vehicles)
- **High Performance**: GPU-accelerated training with 512 parallel environments

## Technology Stack

- **Simulation**: Isaac Sim + Isaac Lab
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: RSL-RL

## Usage
```bash
# Train the model
python scripts/rsl_rl/train.py --task=Template-Navigation-Grid-v0 --num_envs=512

# Run inference
python scripts/rsl_rl/play.py --task=Template-Navigation-Grid-v0
```

## Requirements

- Isaac Lab 5.0+
- Isaac Sim
- Python 3.11
- NVIDIA GPU with CUDA support

## Author

Yahya Hussein Yahya Alsabahi
