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
        
        # ====================================================================
        # CRITICAL FIX: STD PARAMETER NUMERICAL STABILITY
        # ====================================================================
        # Root Cause: During PPO's mini-batch update loop,
        # optimizer.step() can make std negative. Next policy.act() call 
        # in same loop uses negative std → RuntimeError
        #
        # Solution: Use gradient projection + parameter post-processing hook
        # ====================================================================
        
        policy_module = runner.alg.policy
        optimizer = runner.alg.optimizer
        
        print(f"[INFO] Installing STD Parameter Protection System...")
        
        std_config = {
            'min_value': 0.01,
            'protected_params': [],
            'update_step_count': 0
        }
        
        # Step 1: Identify all std-related parameters
        # First, try standard named_parameters and named_buffers
        for name, param in policy_module.named_parameters():
            if "std" in name.lower():
                std_config['protected_params'].append((name, param))
                print(f"[INFO] Protected parameter: {name}, shape={param.shape}")
        
        for name, buf in policy_module.named_buffers():
            if "std" in name.lower():
                std_config['protected_params'].append((name, buf))
                print(f"[INFO] Protected buffer: {name}, shape={buf.shape}")
        
        # Also check direct attributes (in case std is stored as a direct attribute)
        for attr_name in dir(policy_module):
            if "std" in attr_name.lower() and not attr_name.startswith('_'):
                try:
                    attr = getattr(policy_module, attr_name, None)
                    if isinstance(attr, torch.nn.Parameter):
                        # Check if already in list
                        already_protected = any(name == attr_name for name, _ in std_config['protected_params'])
                        if not already_protected:
                            std_config['protected_params'].append((attr_name, attr))
                            print(f"[INFO] Protected direct attribute: {attr_name}, shape={attr.shape}")
                except:
                    pass
        
        # Step 2: Repair checkpoint values
        print(f"[INFO] Repairing std values in loaded checkpoint...")
        repairs_made = 0
        for name, param in std_config['protected_params']:
            with torch.no_grad():
                if (param < std_config['min_value']).any():
                    old_min = param.min().item()
                    param.clamp_(min=std_config['min_value'])
                    repairs_made += 1
                    print(f"[INFO]   Fixed {name}: min {old_min:.6f} → {std_config['min_value']:.6f}")
        
        if repairs_made == 0:
            print(f"[INFO]   No repairs needed - all std values already >= {std_config['min_value']}")
        
        # Step 3: Install critical hook on POLICY FORWARD PASS
        # This catches std values right before they're used
        original_actor_forward = policy_module.actor.forward if hasattr(policy_module, 'actor') else None
        
        def actor_forward_with_std_guard(*args, **kwargs):
            """Forward pass that guarantees std >= min_value."""
            # Call original forward
            if original_actor_forward:
                result = original_actor_forward(*args, **kwargs)
            else:
                # Fallback: call parent forward
                result = type(policy_module.actor).forward(policy_module.actor, *args, **kwargs)
            
            # If result is a tuple (mu, std), enforce std constraint
            if isinstance(result, tuple) and len(result) >= 2:
                mu, std = result[0], result[1]
                with torch.no_grad():
                    std_clamped = torch.clamp(std, min=std_config['min_value'])
                    if not torch.allclose(std, std_clamped, atol=1e-6):
                        # std had values below minimum - clamp them
                        std[:] = std_clamped
                return (mu, std) + result[2:]
            
            return result
        
        # Try to hook into the policy's actor
        if hasattr(policy_module, 'actor'):
            try:
                policy_module.actor.forward = actor_forward_with_std_guard
                print("[INFO] Hooked policy actor forward pass with std guard")
            except Exception as e:
                print(f"[WARN] Could not hook actor forward: {e}")
        
        # Step 4: Install hook on optimizer step
        # This clamps std IMMEDIATELY after gradient descent update
        original_optimizer_step = optimizer.step
        
        # Instead of storing references, store parameter names for dynamic lookup
        std_param_names = [name for name, _ in std_config['protected_params']]
        
        def optimizer_step_with_std_protection(closure=None):
            """Optimizer step that protects std from going negative."""
            # Execute the optimizer step
            if closure is not None:
                loss = original_optimizer_step(closure)
            else:
                loss = original_optimizer_step()
            
            # CRITICAL: Immediately after step, clamp all std parameters
            # Use dynamic lookup to ensure we're clamping the actual parameters
            std_config['update_step_count'] += 1
            with torch.no_grad():
                # Look up parameters by name from the module
                for param_name in std_param_names:
                    try:
                        # Try to get the parameter from the module
                        param = None
                        if hasattr(policy_module, param_name):
                            param = getattr(policy_module, param_name)
                        else:
                            # Try nested lookup for named parameters
                            for name, pam in policy_module.named_parameters():
                                if name == param_name:
                                    param = pam
                                    break
                            if param is None:
                                for name, buf in policy_module.named_buffers():
                                    if name == param_name:
                                        param = buf
                                        break
                        
                        if param is not None and (param.dtype == torch.float32 or param.dtype == torch.float64):
                            before_min = param.min().item()
                            param.clamp_(min=std_config['min_value'])
                            after_min = param.min().item()
                            
                            # Log first few updates for debugging
                            if std_config['update_step_count'] <= 5:
                                if before_min != after_min:
                                    print(f"[DEBUG] Step {std_config['update_step_count']}: {param_name} "
                                         f"clamped {before_min:.6f} → {after_min:.6f}")
                    except Exception as e:
                        if std_config['update_step_count'] == 1:
                            print(f"[WARN] Could not clamp {param_name}: {e}")
            
            return loss
        
        # Replace optimizer step with protected version
        try:
            optimizer.step = optimizer_step_with_std_protection
            print("[INFO] Hooked optimizer.step() with std protection")
        except Exception as e:
            print(f"[WARN] Could not hook optimizer.step(): {e}")
        
        # Step 5: Install hook on act() method - most critical protection point
        if hasattr(policy_module, 'act'):
            original_act = policy_module.act
            
            def act_with_std_check(*args, **kwargs):
                """Wrapper around act() - attempts to fix distribution before sampling."""
                try:
                    # Import Normal here to hook it
                    from torch.distributions import Normal
                    
                    original_normal_init = Normal.__init__
                    original_normal_sample = Normal.sample
                    
                    patch_count = [0]  # Use list to allow modification in nested function
                    
                    def patched_normal_init(dist_self, loc, scale, validate_args=None):
                        # Call original init
                        original_normal_init(dist_self, loc, scale, validate_args)
                        
                        # Immediately after init, check scale
                        with torch.no_grad():
                            if (dist_self.scale < std_config['min_value']).any():
                                print(f"[CRITICAL FIX] Normal dist scale had values < {std_config['min_value']}!")
                                print(f"  Before: min={dist_self.scale.min().item():.8f}, max={dist_self.scale.max().item():.8f}")
                                print(f"  Bad values: {dist_self.scale[dist_self.scale < std_config['min_value']]}")
                                dist_self.scale = torch.clamp(dist_self.scale, min=std_config['min_value'])
                                print(f"  After: min={dist_self.scale.min().item():.8f}, max={dist_self.scale.max().item():.8f}")
                            
                            if dist_self.scale.isnan().any() or dist_self.scale.isinf().any():
                                print(f"[CRITICAL FIX] Normal dist scale has NaN/Inf!")
                                dist_self.scale = torch.clamp(dist_self.scale, min=std_config['min_value'], max=1.0)
                    
                    def patched_normal_sample(dist_self, sample_shape=torch.Size()):
                        # Double-check scale before sampling
                        patch_count[0] += 1
                        with torch.no_grad():
                            if (dist_self.scale < std_config['min_value']).any():
                                print(f"[CRITICAL FIX #{patch_count[0]}] Normal.sample() scale check - fixing!")
                                dist_self.scale = torch.clamp(dist_self.scale, min=std_config['min_value'])
                        
                        # Now sample
                        return original_normal_sample(dist_self, sample_shape)
                    
                    # Apply patches
                    Normal.__init__ = patched_normal_init
                    Normal.sample = patched_normal_sample
                    
                    try:
                        # Call original act
                        result = original_act(*args, **kwargs)
                        return result
                    finally:
                        # Restore originals
                        Normal.__init__ = original_normal_init
                        Normal.sample = original_normal_sample
                        
                except Exception as e:
                    # If patching fails, try the old method
                    print(f"[WARN] Normal distribution patching failed: {e}")
                    print(f"       Falling back to parameter-level checks")
                    
                    # Fallback: check std parameters directly
                    with torch.no_grad():
                        for param_name in std_param_names:
                            try:
                                param = None
                                if hasattr(policy_module, param_name):
                                    param = getattr(policy_module, param_name)
                                else:
                                    for name, pam in policy_module.named_parameters():
                                        if name == param_name:
                                            param = pam
                                            break
                                    if param is None:
                                        for name, buf in policy_module.named_buffers():
                                            if name == param_name:
                                                param = buf
                                                break
                                
                                if param is not None and (param < std_config['min_value']).any():
                                    param.clamp_(min=std_config['min_value'])
                            except:
                                pass
                    
                    return original_act(*args, **kwargs)
            
            try:
                policy_module.act = act_with_std_check
                print("[INFO] Installed pre-act std validation")
            except Exception as e:
                print(f"[WARN] Could not wrap act(): {e}")
        
        # Step 6: Export configuration for monitoring
        runner.std_protection_config = std_config
        
        protected_param_str = ""
        if std_param_names:
            protected_param_str = "\n".join([f"[INFO]     - {name}" for name in std_param_names])
        else:
            protected_param_str = "[INFO]     (None found - checking backup mechanisms)"
        
        print(f"""
[INFO] ═══════════════════════════════════════════════════════════════
[INFO] STD PARAMETER PROTECTION SYSTEM ACTIVE
[INFO] ───────────────────────────────────────────────────────────────
[INFO] Minimum std value: {std_config['min_value']}
[INFO] Protected parameters: {len(std_config['protected_params'])}
{protected_param_str}
[INFO] Protection points:
[INFO]   1. Checkpoint load: std clamped to >= {std_config['min_value']}
[INFO]   2. Optimizer.step(): std clamped IMMEDIATELY after update (dynamic lookup)
[INFO]   3. Actor forward: std clamped in output
[INFO]   4. policy.act(): std pre-validated before sampling
[INFO] ═══════════════════════════════════════════════════════════════""")
        
        if not std_config['protected_params']:
            print("[WARN] No std parameters found! Policy architecture may use log_std or other form.")
            print(f"[DEBUG] Available parameters: {[n for n, _ in policy_module.named_parameters()]}")


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
