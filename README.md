# YHBot Navigation

Autonomous warehouse robot navigation using Deep Reinforcement Learning with NVIDIA Isaac Lab.

## Overview

YHBot is an Embodied AI system that learns to navigate complex warehouse environments through curriculum learning and Deep Reinforcement Learning. The robot achieves 95% navigation success in a full warehouse, and dynamic obstacles.

This project demonstrates the power of GPU-accelerated simulation for training physical AI systems, achieving learning rates far beyond real-time through massively parallel training.

## Demo

![YHBot Warehouse Navigation](assets/demo.gif)

## Key Features

- **Curriculum Learning**: Progressive training across 3 phases from simple to complex environments
- **Deep RL Navigation**: PPO-based policy learning for adaptive real-time navigation
- **Sensor Fusion**: Integration of 2D LiDAR and contact sensors (71-dimensional observation space)
- **High Performance**: GPU-accelerated training with 8,192 parallel environments
- **Dynamic Obstacles**: Handles static and moving objects in warehouse settings
- **Robust Performance**: 95% success rate in complex navigation scenarios

## Training Phases

### Phase 1: Basic Navigation
- **Environment**: Open grid space (60m × 80m)
- **Objective**: Learn goal-reaching behavior
- **Focus**: Foundation movement and goal orientation

### Phase 2: Obstacle Avoidance
- **Environment**: Grid with static obstacles
- **Objective**: Static obstacle avoidance with collision detection
- **Focus**: LiDAR-based perception and collision avoidance

### Phase 3: Warehouse Mastery
- **Environment**: Realistic warehouse simulation (31m × 54m)
- **Objective**: Navigate complex aisles and racks
- **Layout**: 7 parallel racks, 6 aisles, 3 static + 2 dynamic obstacles
- **Focus**: Real-world preparation and complex path planning

## Architecture

### Robot Platform
- **Type**: 2-wheel differential drive robot
- **Control**: Direct wheel velocity control (left & right middle wheels)
- **Sensors**: 
  - 2D LiDAR (180° FOV, 60 rays, 10m range)
  - Contact sensors (chassis and roof)
  - Velocity sensors (linear & angular)

### Deep RL Configuration
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Framework**: SKRL + NVIDIA Isaac Lab
- **Neural Network**: [256, 256, 128] layers with ELU activation (~235K parameters)
- **Training**: 8M timesteps across 8,192 parallel environments
- **Observation Space**: 71 dimensions (LiDAR + velocities + goal info + action history)
- **Action Space**: 2D continuous (left/right wheel velocities)

### Reward Structure
- **Goal Reached**: +500 (threshold: 0.6m)
- **Progress to Goal**: +10
- **Heading Alignment**: +0.3
- **Collision**: -100 (force threshold: 0.5N)
- **Alive Penalty**: -0.01
- **Action Smoothness**: -0.01

## Technology Stack

- **Simulation**: NVIDIA Isaac Sim + Isaac Lab
- **Deep RL**: Proximal Policy Optimization (PPO)
- **Framework**: SKRL
- **GPU Acceleration**: CUDA
- **Python**: 3.10+

## Installation

### Prerequisites
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA support (RTX 3000 series or higher recommended)
- Ubuntu 22.04 or 24.04

### Setup Steps

#### 1. Install Miniconda (if not already installed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

#### 2. Accept Conda Terms & Install System Dependencies
```bash
conda config --set solver libmamba
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

sudo apt-get update && sudo apt-get install -y git wget build-essential cmake libglu1-mesa
```

#### 3. Create Python Environment & Install PyTorch
```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab

pip install --upgrade pip
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
```

#### 4. Install Isaac Sim & Isaac Lab
```bash
pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com

git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```
*Accept EULA when prompted: Type "Yes"*

#### 5. Clone and Install YHBot Project
```bash
# Clone the repository
git clone https://github.com/[your-username]/yhbot_navigation.git
cd yhbot_navigation

# Navigate to source directory and install

cd /root/yhbot_navigation/source/yhbot_navigation
pip install -e .
```

#### 6. Verify Installation
```bash
# List available environments
python -c "import gymnasium as gym; print([env for env in gym.envs.registry if 'YHBot' in env])"

# Expected output:
# ['Isaac-Navigation-Grid-YHBot-v0', 'Isaac-Navigation-WarehouseGrid-YHBot-v0', ...]
```

#### 7. Fix Paths
```bash
cd /root/yhbot_navigation
find . -type f -name "*.py" -exec sed -i 's|/home/yahya/Documents/fyp/yhbot_navigation|/root/yhbot_navigation|g' {} +
find . -type f -name "*.py" -exec sed -i 's|/home/yahya|/root|g' {} +
find . -type f -name "*.yaml" -exec sed -i 's|/home/yahya/Documents/fyp/yhbot_navigation|/root/yhbot_navigation|g' {} +
find . -type f -name "*.yaml" -exec sed -i 's|/home/yahya|/root|g' {} +
```


## Usage

### Training

**Phase 1: Grid Environment**
```bash
python scripts/skrl/train.py \
    --task Isaac-Navigation-Grid-YHBot-v0 \
    --num_envs 8192 \
    --headless
```

**Phase 3: Warehouse Grid Environment**
```bash
python scripts/skrl/train.py \
    --task Isaac-Navigation-WarehouseGrid-YHBot-v0 \
    --num_envs 8192 \
    --headless
```

### Inference

**Run trained policy (single environment)**
```bash
python scripts/skrl/play.py \
    --task Isaac-Navigation-WarehouseGrid-YHBot-Play-v0 \
    --checkpoint /path/to/model.pt
```

## Author

**Yahya Hussein Yahya Alsabahi**
- LinkedIn: https://www.linkedin.com/in/ya7ya-hussein
- GitHub: https://github.com/ya7ya-hussein
- Email: ya7ya-hussein@outlook.com