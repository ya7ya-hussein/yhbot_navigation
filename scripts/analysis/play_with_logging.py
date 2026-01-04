# play_with_logging.py
"""
Play a trained YHBot agent with comprehensive behavior logging.
Saves detailed metrics to CSV for model analysis.

TEMPORARY FILE - Delete after analysis is complete.

Setup:
    1. Create directory: mkdir -p ~/Documents/fyp/yhbot_navigation/scripts/analysis
    2. Copy files:
       cp behavior_logger.py ~/Documents/fyp/yhbot_navigation/scripts/analysis/
       cp play_with_logging.py ~/Documents/fyp/yhbot_navigation/scripts/analysis/

Usage:
    cd ~/Documents/fyp/yhbot_navigation
    python scripts/analysis/play_with_logging.py --task Isaac-Navigation-Grid-YHBot-v0 --num_envs 1

    # With custom episode limit:
    python scripts/analysis/play_with_logging.py --task Isaac-Navigation-Grid-YHBot-v0 --num_envs 1 --max_episodes 50

    # With custom output directory:
    python scripts/analysis/play_with_logging.py --task Isaac-Navigation-Grid-YHBot-v0 --num_envs 1 --output_dir my_analysis

Output:
    CSV file in: ~/Documents/fyp/yhbot_navigation/analysis_logs/yhbot_behavior_log_YYYYMMDD_HHMMSS.csv
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained agent with behavior logging.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
parser.add_argument("--video_length", type=int, default=200, help="Video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (use 1 for logging).")
parser.add_argument("--task", type=str, default=None, help="Task name.")
parser.add_argument("--agent", type=str, default=None, help="Agent config entry point.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use pretrained checkpoint.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")

# Logging-specific arguments
parser.add_argument("--max_episodes", type=int, default=100, help="Maximum episodes to log (0 = unlimited).")
parser.add_argument("--max_timesteps", type=int, default=0, help="Maximum timesteps to log (0 = unlimited).")
parser.add_argument("--output_dir", type=str, default="analysis_logs", help="Output directory for CSV files.")
parser.add_argument("--env_index", type=int, default=0, help="Environment index to log.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after Isaac Sim is initialized."""

import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

# Check skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Install skrl>={SKRL_VERSION}")
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import yhbot_navigation.tasks  # noqa: F401

# Import behavior logger from same directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from behavior_logger import BehaviorLogger

# Config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with behavior logging."""
    
    # Task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # Override configurations
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ML framework config
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # Seed
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # Checkpoint path
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Pre-trained checkpoint unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    unwrapped_env = env.unwrapped

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Get step dt for logger
    try:
        step_dt = env.step_dt
    except AttributeError:
        step_dt = env.unwrapped.step_dt

    # Initialize behavior logger
    output_dir = os.path.abspath(args_cli.output_dir)
    logger = BehaviorLogger(
        output_dir=output_dir,
        env_index=args_cli.env_index,
        step_dt=step_dt
    )
    
    print(f"\n{'='*60}")
    print("BEHAVIOR LOGGING ENABLED")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Max episodes: {args_cli.max_episodes if args_cli.max_episodes > 0 else 'unlimited'}")
    print(f"  Max timesteps: {args_cli.max_timesteps if args_cli.max_timesteps > 0 else 'unlimited'}")
    print(f"  Step dt: {step_dt:.4f}s")
    print(f"{'='*60}\n")

    # Video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # Configure runner
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    # Load checkpoint
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # Reset environment
    obs, _ = env.reset()
    timestep = 0
    total_timesteps = 0
    last_episode = -1
    
    # Tracking for progress display
    start_time = time.time()
    last_status_time = start_time

    try:
        while simulation_app.is_running():
            loop_start = time.time()

            with torch.inference_mode():
                # Agent action
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                
                # Environment step
                obs, _, _, _, _ = env.step(actions)
            
            # Log timestep
            data = logger.log_timestep(unwrapped_env, actions)
            total_timesteps += 1
            
            # Check episode change
            if data.episode_number != last_episode:
                if last_episode >= 0:
                    print(f"[LOG] Episode {last_episode} ended: {data.termination_reason}")
                last_episode = data.episode_number
            
            # Progress display every 5 seconds
            current_time = time.time()
            if current_time - last_status_time >= 5.0:
                elapsed = current_time - start_time
                stats = logger.get_stats()
                print(f"[STATUS] Episodes: {stats['current_episode']+1} | "
                      f"Timesteps: {total_timesteps} | "
                      f"Elapsed: {elapsed:.1f}s | "
                      f"Rate: {total_timesteps/elapsed:.1f} steps/s")
                last_status_time = current_time
            
            # Check termination conditions
            if args_cli.max_episodes > 0 and data.episode_number >= args_cli.max_episodes:
                print(f"\n[INFO] Reached max episodes ({args_cli.max_episodes})")
                break
            
            if args_cli.max_timesteps > 0 and total_timesteps >= args_cli.max_timesteps:
                print(f"\n[INFO] Reached max timesteps ({args_cli.max_timesteps})")
                break
            
            # Video recording
            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            # Real-time delay
            if args_cli.real_time:
                sleep_time = step_dt - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        # Close logger and get filepath
        csv_path = logger.close()
        
        # Final summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("LOGGING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total episodes: {logger._current_episode + 1}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Average rate: {total_timesteps/elapsed:.1f} steps/s")
        print(f"  CSV file: {csv_path}")
        print(f"{'='*60}\n")
        
        # Close environment
        try:
            env.close()
        except Exception as e:
            print(f"[WARN] Environment cleanup error (safe to ignore): {e}")


if __name__ == "__main__":
    main()
    simulation_app.close()