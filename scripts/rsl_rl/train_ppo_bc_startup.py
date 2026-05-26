# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--bc_model_path", type=str, default=None, help="Path to pre-trained Behavioral Cloning model (.pt) to initialize Actor.")
parser.add_argument("--critic_warmup_iters", type=int, default=500, help="Number of iterations to freeze Actor and only train Critic.")

parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import logging
import os
import torch
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import pickup_place_direct_0203.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # ========================== BC WEIGHT TRANSFER & CRITIC WARMUP ==========================
    if args_cli.bc_model_path:
        print(f"\n[INFO] Loading BC Pretrained Model from: {args_cli.bc_model_path}")
        bc_state_dict = torch.load(args_cli.bc_model_path, map_location=env.unwrapped.device)
        
        # 1. Load Normalizers and Vision Encoders into Environment
        if hasattr(env.unwrapped, "load_bc_normalization_and_encoders"):
            env.unwrapped.load_bc_normalization_and_encoders(bc_state_dict)
        else:
            print("[WARN] Environment does not support load_bc_normalization_and_encoders! (Is it 0313 or later?)")
            
        # 2. Extract MLP weights from BC StudentActor and map to RSL-RL Actor
        print("[INFO] Mapping BC MLP weights to RSL-RL Actor...")
        
        # Based on runtime inspection, for ActorCritic:
        # runner.alg.policy is the ActorCritic module
        # runner.alg.policy.actor is the MLPModel for the policy
        target_policy = runner.alg.policy.actor

        # BC weights start with 'mlp.' (e.g., 'mlp.0.weight')
        # Target actor keys are direct (e.g., '0.weight')
        new_actor_dict = {}
        for k, v in bc_state_dict.items():
            if k.startswith("mlp."):
                new_key = k.replace("mlp.", "") # Strip 'mlp.' prefix
                new_actor_dict[new_key] = v
        
        try:
            missing, unexpected = target_policy.load_state_dict(new_actor_dict, strict=False)
            print(f"[INFO] Injected BC Actor weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            # --- FINAL NOISE DEFENSE ---
            # Manually freeze Action Noise parameters to keep them at the desired init_noise_std (e.g. 0.1)
            # This prevents PPO from trying to increase randomness via Entropy Bonus.
            if hasattr(target_policy, 'std'):
                target_policy.std.requires_grad = False
                print("[INFO] Action Noise (std) frozen.")
            if hasattr(target_policy, 'log_std'):
                target_policy.log_std.requires_grad = False
                print("[INFO] Action Noise (log_std) frozen.")
                
        except Exception as e:
            print(f"[ERROR] Failed to load BC Actor weights: {e}")

        # 3. Setup Critic Warmup and Exploration Logging
        original_update = runner.alg.update
        
        warmup_iters = args_cli.critic_warmup_iters
        print(f"[INFO] Initializing Critic Warmup strategy for the first {warmup_iters} iterations.")
        current_iter = [0] # List reference to modify inside inner function
        
        # Optimization: capture the actor parameter list once from the resolved model
        actor_params = list(target_policy.parameters())
        # Access the policy root for std
        root_policy = runner.alg.policy
        
        def warmup_update(*args, **kwargs):
            iter_count = current_iter[0]
            
            if iter_count < warmup_iters:
                if iter_count == 0:
                    print(f"\n[INFO] >>> CRITIC WARMUP STARTED ({warmup_iters} iters) <<<", flush=True)
                
                # Freeze actor parameters
                for param in actor_params:
                    param.requires_grad = False
                    
                # Call original PPO update
                result = original_update(*args, **kwargs)
                
                # Unfreeze for next potentially normal update
                for param in actor_params:
                    param.requires_grad = True
                    
                if iter_count % 50 == 0 or iter_count == warmup_iters - 1:
                    print(f"  [Warmup] Completed Iteration {iter_count}/{warmup_iters}.", flush=True)
                
                if iter_count == warmup_iters - 1:
                    print(f"[INFO] >>> CRITIC WARMUP FINISHED. Joint training begins. <<<\n", flush=True)
            else:
                result = original_update(*args, **kwargs)
            
            # Print exploration std periodically
            if iter_count % 100 == 0:
                # Robustly find std based on ActorCritic structure
                std_val = getattr(root_policy, "std", None)
                if std_val is None:
                    std_val = getattr(root_policy, "action_std", None)
                
                if std_val is not None:
                    print(f"  [Exploration] Current Actor Action Std: {std_val.mean().item():.4f}", flush=True)
                
            current_iter[0] += 1
            return result
            
        # Hook the modified update method into the algorithm
        runner.alg.update = warmup_update
    # ========================================================================================

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
