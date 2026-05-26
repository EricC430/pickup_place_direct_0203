# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to verify if the Teacher model is performing correctly in the Distillation environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import time
import torch

from isaaclab.app import AppLauncher

# local imports (Ensure cli_args is available in the search path)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "rsl_rl")))
import cli_args # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Verify Teacher Model in Distillation Environment.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during verification.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Pickup-Place-Direct-0421-Static-Distill-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Check for checkpoint since it's required for verification
if args_cli.checkpoint is None:
    print("[ERROR] Please provide a checkpoint path using --checkpoint.")
    sys.exit(1)

# default to headless if video recording is requested for verify
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runners import DistillationRunner
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.dict import print_dict

import pickup_place_direct_0203.tasks # noqa: F401

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    # Override configs
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Wrap for video recording
    if args_cli.video:
        video_dir = os.path.join("logs", "verify_teacher", "videos")
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Saving verification video to: {video_dir}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading Teacher Model architecture and weights...")
    # Setup the runner and load weights
    # We use log_dir=None because we don't want to create training logs
    runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # Force load into the teacher part
    # DistillationRunner.load will call policy.load_state_dict
    runner.load(args_cli.checkpoint)
    
    # Extract the policy module
    policy_nn = runner.alg.policy
    policy_nn.eval()

    print("\033[1;32m[Diagnostic] Verification Loop Started. FORCING TEACHER ACTIONS.\033[0m")
    
    # Reset environment
    obs = env.get_observations()
    
    # Simulate environment
    step_count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # -------------------------------------------------------------
            # CRITICAL: We bypass student actions and call the Teacher!
            # -------------------------------------------------------------
            # obs is a dict {"policy": ..., "teacher": ...} if RslRlVecEnvWrapper 
            # is used with a Distillation configuration.
            actions = policy_nn.evaluate(obs)
            
            # Env stepping
            obs, _, dones, extras = env.step(actions)
            
            # Handle recurrent state reset if any (though Teacher is likely MLP)
            policy_nn.reset(dones)
            
        step_count += 1
        if args_cli.video and step_count >= args_cli.video_length:
            print(f"[INFO] Reached requested video length ({args_cli.video_length}). Exiting.")
            break

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
