# scripts/skrl/play_with_bridge.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl with dashboard bridge support.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import carb.input

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# Bridge arguments
parser.add_argument("--bridge_url", type=str, default="http://localhost:3001", help="Dashboard backend URL.")
parser.add_argument("--bridge_hz", type=float, default=30.0, help="Bridge emission frequency.")
parser.add_argument("--bridge_env_index", type=int, default=0, help="Environment index to monitor.")
parser.add_argument("--disable_bridge", action="store_true", default=False, help="Disable dashboard bridge.")

# Frame streaming arguments
parser.add_argument("--enable_frame_stream", action="store_true", default=False, help="Enable frame streaming to dashboard.")
parser.add_argument("--frame_width", type=int, default=1280, help="Frame width for streaming (default: 1280 for 16:9).")
parser.add_argument("--frame_height", type=int, default=720, help="Frame height for streaming (default: 720 for 16:9).")
parser.add_argument("--frame_fps", type=float, default=15.0, help="Target FPS for frame streaming.")
parser.add_argument("--frame_quality", type=int, default=75, help="JPEG quality for frame streaming (1-100).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Enable cameras/rendering if video recording OR frame streaming is enabled
if args_cli.video or args_cli.enable_frame_stream:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
import time
import torch

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
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

from isaaclab.sim import SimulationContext

# Import bridge components
bridge_path = os.path.join(os.path.dirname(__file__), '..', '..', 'isaaclab_bridge')
sys.path.insert(0, bridge_path)
from bridge_main import IsaacLabBridge

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Determine render mode: enable rgb_array if video OR frame streaming is enabled
    render_mode = "rgb_array" if (args_cli.video or args_cli.enable_frame_stream) else None
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)
    
    # Store reference to unwrapped env for frame capture
    unwrapped_env = env.unwrapped

    is_paused = False
    
    import omni.appwindow
    
    def on_keyboard_event(event, *args, **kwargs):
        nonlocal is_paused
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.P:
                is_paused = not is_paused
                print(f"[INFO] Simulation {'PAUSED' if is_paused else 'RESUMED'} (Press P to toggle)")
                return True
        return False
    
    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    keyboard_sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(keyboard, on_keyboard_event)


    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Initialize bridge
    bridge = None
    if not args_cli.disable_bridge:
        print(f"[INFO] Initializing dashboard bridge...")
        bridge = IsaacLabBridge(
            server_url=args_cli.bridge_url,
            target_hz=args_cli.bridge_hz,
            env_index=args_cli.bridge_env_index,
            persist_data=False,
            frame_width=args_cli.frame_width,
            frame_height=args_cli.frame_height,
            frame_fps=args_cli.frame_fps,
            frame_quality=args_cli.frame_quality
        )
        bridge.register_environment(unwrapped_env)
        
        # Disable frame capture if not explicitly enabled
        if not args_cli.enable_frame_stream:
            bridge.set_frame_capture_enabled(False)
        
        if bridge.start():
            print(f"[INFO] Bridge started successfully")
            print(f"[INFO]   Server: {args_cli.bridge_url}")
            print(f"[INFO]   Frequency: {args_cli.bridge_hz} Hz")
            print(f"[INFO]   Monitoring env: {args_cli.bridge_env_index}")
            if args_cli.enable_frame_stream:
                print(f"[INFO]   Frame streaming: {args_cli.frame_width}x{args_cli.frame_height} @ {args_cli.frame_fps} FPS")
        else:
            print(f"[WARN] Failed to start bridge - continuing without dashboard")
            bridge = None

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    frame_count = 0

    
    # simulate environment
    try:
        while simulation_app.is_running():

            if is_paused:
                simulation_app.update()
                time.sleep(0.016)
                continue

            start_time = time.time()

            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                outputs = runner.agent.act(obs, timestep=0, timesteps=0)
                # - multi-agent (deterministic) actions
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                # - single-agent (deterministic) actions
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
                # env stepping
                obs, _, _, _, _ = env.step(actions)
            
            # Capture and stream frame (if enabled)
            # This runs on the main thread and respects the configured FPS
            if bridge is not None and args_cli.enable_frame_stream:
                if bridge.capture_and_submit_frame(unwrapped_env):
                    frame_count += 1
            
            if args_cli.video:
                timestep += 1
                # exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)


    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    
    finally:
        
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
        
        # Stop bridge
        if bridge is not None:
            print("[INFO] Stopping bridge...")
            if args_cli.enable_frame_stream:
                print(f"[INFO] Total frames streamed: {frame_count}")
            bridge.stop()
        
        # close the simulator
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()