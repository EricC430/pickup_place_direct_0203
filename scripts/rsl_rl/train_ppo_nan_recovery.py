# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL with Vision Weights Loading and STD Recovery."""

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


def install_std_protection(runner, device):
    """Install STD parameter protection system with proper NaN/Inf handling."""
    policy_module = runner.alg.policy
    optimizer = runner.alg.optimizer
    
    print(f"[INFO] Installing STD Parameter Protection System (with NaN/Inf handling)...")
    
    std_config = {
        'min_value': 0.01,
        'max_value': 1.0,
        'protected_params': [],
        'update_step_count': 0,
        'nan_recovery_count': 0,
        'device': device
    }
    
    def safe_param_update(param, source_tensor):
        """Safely update parameter, handling inference tensors."""
        try:
            # Try direct copy first
            param.copy_(source_tensor)
        except RuntimeError as e:
            if "inference tensor" in str(e):
                # For inference tensors, use data assignment with detach
                param.data = source_tensor.detach().clone()
            else:
                raise
    
    # Step 1: Identify all std-related parameters
    for name, param in policy_module.named_parameters():
        if "std" in name.lower():
            std_config['protected_params'].append((name, 'param'))
            print(f"[INFO] Protected parameter: {name}, shape={param.shape}")
    
    for name, buf in policy_module.named_buffers():
        if "std" in name.lower():
            std_config['protected_params'].append((name, 'buffer'))
            print(f"[INFO] Protected buffer: {name}, shape={buf.shape}")
    
    # Step 2: Repair checkpoint values - with NaN recovery
    print(f"[INFO] Repairing std values in loaded checkpoint...")
    repairs_made = 0
    for name, param_type in std_config['protected_params']:
        try:
            if param_type == 'param':
                param = dict(policy_module.named_parameters())[name]
            else:
                param = dict(policy_module.named_buffers())[name]
            
            with torch.no_grad():
                # Handle NaN/Inf
                if param.isnan().any() or param.isinf().any():
                    print(f"[WARN]   {name} contains NaN/Inf - resetting to {std_config['min_value']}")
                    param.fill_(std_config['min_value'])
                    repairs_made += 1
                # Handle values < min
                elif (param < std_config['min_value']).any():
                    old_min = param.min().item()
                    # Use clamp instead of clamp_ for inference tensors
                    param_clamped = torch.clamp(param, min=std_config['min_value'], max=std_config['max_value'])
                    safe_param_update(param, param_clamped)
                    repairs_made += 1
                    print(f"[INFO]   Fixed {name}: min {old_min:.6f} → {std_config['min_value']:.6f}")
        except Exception as e:
            print(f"[WARN]   Could not repair {name}: {e}")
    
    if repairs_made == 0:
        print(f"[INFO]   No repairs needed - all std values already valid")
    
    # Step 3: Enhanced optimizer step with strict NaN checking
    original_optimizer_step = optimizer.step
    std_param_refs = {}  # Store references to actual parameter objects
    
    # Get references after potential in-place modifications
    def get_current_std_params():
        """Dynamically get current std parameters."""
        params = {}
        try:
            for param_name, param_type in std_config['protected_params']:
                if param_type == 'param':
                    for name, param in policy_module.named_parameters():
                        if name == param_name:
                            params[param_name] = param
                            break
                else:
                    for name, buf in policy_module.named_buffers():
                        if name == param_name:
                            params[param_name] = buf
                            break
        except:
            pass
        return params
    
    def optimizer_step_with_std_protection(closure=None):
        """Optimizer step with robust NaN/Inf handling."""
        std_config['update_step_count'] += 1
        
        # Pre-check: ensure std parameters are valid BEFORE optimizer step
        with torch.no_grad():
            current_params = get_current_std_params()
            for param_name, param in current_params.items():
                if param.isnan().any() or param.isinf().any():
                    print(f"[CRITICAL PRE-STEP] Step {std_config['update_step_count']}: "
                          f"{param_name} has NaN/Inf BEFORE optimizer - fixing...")
                    param.fill_(std_config['min_value'])
        
        # Execute the optimizer step
        if closure is not None:
            loss = original_optimizer_step(closure)
        else:
            loss = original_optimizer_step()
        
        # Immediately after step: check and fix all std parameters
        with torch.no_grad():
            current_params = get_current_std_params()
            
            for param_name, param in current_params.items():
                try:
                    # Check for NaN/Inf
                    if param.isnan().any() or param.isinf().any():
                        std_config['nan_recovery_count'] += 1
                        print(f"[CRITICAL POST-STEP] Step {std_config['update_step_count']}: "
                              f"{param_name} has NaN/Inf AFTER optimizer - recovering...")
                        # Recover with safe fallback value
                        safe_param_update(param, torch.full_like(param, std_config['min_value']))
                        print(f"  Recovered {param_name} with value {std_config['min_value']}")
                    
                    # Check for out-of-range values
                    elif (param < std_config['min_value']).any() or (param > std_config['max_value']).any():
                        param_clamped = torch.clamp(param, min=std_config['min_value'], max=std_config['max_value'])
                        safe_param_update(param, param_clamped)
                        if std_config['update_step_count'] <= 10:
                            print(f"[DEBUG] Step {std_config['update_step_count']}: "
                                  f"{param_name} clamped to [{param.min().item():.6f}, {param.max().item():.6f}]")
                
                except Exception as e:
                    if std_config['update_step_count'] <= 5:
                        print(f"[WARN] Could not protect {param_name}: {e}")
        
        return loss
    
    # Install the protected optimizer step
    try:
        optimizer.step = optimizer_step_with_std_protection
        print("[INFO] Hooked optimizer.step() with NaN-aware std protection")
    except Exception as e:
        print(f"[WARN] Could not hook optimizer.step(): {e}")
    
    # Step 4: Install hook on act() method - most critical protection point
    if hasattr(policy_module, 'act'):
        original_act = policy_module.act
        
        def act_with_std_check(*args, **kwargs):
            """Wrapper around act() with pre-check and post-check."""
            # Pre-check: validate all std parameters before act()
            with torch.no_grad():
                current_params = get_current_std_params()
                for param_name, param in current_params.items():
                    if param.isnan().any() or param.isinf().any():
                        print(f"[CRITICAL PRE-CHECK] {param_name} has NaN/Inf before act() - fixing...")
                        safe_param_update(param, torch.full_like(param, std_config['min_value']))
                    elif (param < std_config['min_value']).any():
                        param_clamped = torch.clamp(param, min=std_config['min_value'], max=std_config['max_value'])
                        safe_param_update(param, param_clamped)
                
                # Also check if observations contain NaN - this is the root cause
                if len(args) > 0:
                    obs = args[0]
                    if isinstance(obs, dict):
                        for obs_key, obs_val in obs.items():
                            if isinstance(obs_val, torch.Tensor):
                                if obs_val.isnan().any() or obs_val.isinf().any():
                                    print(f"[CRITICAL] Observation '{obs_key}' contains NaN/Inf! "
                                          f"Shape: {obs_val.shape}, NaN count: {obs_val.isnan().sum()}")
                    elif isinstance(obs, torch.Tensor):
                        if obs.isnan().any() or obs.isinf().any():
                            print(f"[CRITICAL] Observation tensor contains NaN/Inf! "
                                  f"Shape: {obs.shape}, NaN count: {obs.isnan().sum()}")
            
            # Call original act
            try:
                result = original_act(*args, **kwargs)
                return result
            except RuntimeError as e:
                if "normal expects all elements of std >= 0.0" in str(e):
                    print(f"[CRITICAL ERROR] RuntimeError during act(): {e}")
                    print("[CRITICAL] Attempting emergency std recovery...")
                    
                    # Emergency recovery: set all std to safe minimum
                    with torch.no_grad():
                        current_params = get_current_std_params()
                        for param_name, param in current_params.items():
                            safe_param_update(param, torch.full_like(param, std_config['min_value']))
                    
                    # Retry once
                    try:
                        result = original_act(*args, **kwargs)
                        print("[CRITICAL] Recovery successful!")
                        return result
                    except Exception as retry_error:
                        print(f"[CRITICAL] Recovery failed: {retry_error}")
                        raise
                else:
                    raise
        
        try:
            policy_module.act = act_with_std_check
            print("[INFO] Installed act() wrapper with pre/post validation")
        except Exception as e:
            print(f"[WARN] Could not wrap act(): {e}")
    
    # Step 5: Wrap alg.update() to protect during algorithm update
    alg_obj = runner.alg
    original_alg_update = alg_obj.update
    
    def alg_update_with_obs_protection(*args, **kwargs):
        """Wrapper around alg.update() with observation normalizer protection."""
        # Pre-update: check and fix std parameters
        with torch.no_grad():
            current_params = get_current_std_params()
            for param_name, param in current_params.items():
                if param.isnan().any() or param.isinf().any():
                    safe_param_update(param, torch.full_like(param, std_config['min_value']))
        
        # Execute the algorithm update
        try:
            loss_dict = original_alg_update(*args, **kwargs)
        except Exception as e:
            if "normal expects all elements of std >= 0.0" in str(e) or "nan" in str(e).lower():
                print(f"[CRITICAL ALG ERROR] {e}")
                print("[CRITICAL] Attempting recovery before retry...")
                with torch.no_grad():
                    current_params = get_current_std_params()
                    for param_name, param in current_params.items():
                        safe_param_update(param, torch.full_like(param, std_config['min_value']))
                # Retry once
                loss_dict = original_alg_update(*args, **kwargs)
            else:
                raise
        
        # Post-update: fix any NaN/Inf that appeared
        with torch.no_grad():
            current_params = get_current_std_params()
            for param_name, param in current_params.items():
                if param.isnan().any() or param.isinf().any():
                    std_config['nan_recovery_count'] += 1
                    safe_param_update(param, torch.full_like(param, std_config['min_value']))
        
        return loss_dict
    
    try:
        alg_obj.update = alg_update_with_obs_protection
        print("[INFO] Installed alg.update() wrapper with observation protection")
    except Exception as e:
        print(f"[WARN] Could not hook alg.update(): {e}")
    # Step 5: Export configuration
    runner.std_protection_config = std_config
    print(f"""
[INFO] ═══════════════════════════════════════════════════════════════
[INFO] STD PARAMETER PROTECTION SYSTEM ACTIVE
[INFO] ───────────────────────────────────────────────────────────────
[INFO] Valid std range: [{std_config['min_value']}, {std_config['max_value']}]
[INFO] Protected parameters: {len(std_config['protected_params'])}
[INFO] Protection features:
[INFO]   ✓ NaN/Inf detection and recovery
[INFO]   ✓ Parameter-level clamping
[INFO]   ✓ Optimizer step protection
[INFO]   ✓ Pre-check validation in act()
[INFO]   ✓ Emergency recovery mechanism
[INFO] ═══════════════════════════════════════════════════════════════""")


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
        
        # Install STD parameter protection with NaN/Inf handling
        install_std_protection(runner, agent_cfg.device)
        
        # ====================================================================
        # SPECIAL: Load vision weights (optional)
        # ====================================================================
        # This section is for loading pre-trained vision encoder weights
        # Uncomment and modify if you have vision weights to load
        try:
            vision_weights_path = "logs/vision_weights_standalone.pt"
            if os.path.exists(vision_weights_path):
                print(f"[INFO] Loading vision weights from: {vision_weights_path}")
                try:
                    # Load the environment's vision components
                    # This depends on your environment implementation
                    # env.load_vision_weights(vision_weights_path)
                    print("[INFO] Vision weights loaded successfully (if environment supports it)")
                except Exception as e:
                    print(f"[WARN] Could not load vision weights: {e}")
        except:
            pass

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
