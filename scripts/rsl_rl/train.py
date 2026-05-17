# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL. (Clean version - no std protection)"""

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
# --- NEW: BC and Modular Encoder Training Args ---
parser.add_argument("--use_bc_weights", action="store_true", default=False, help="Load BC weights into vision encoders/normalizers.")
parser.add_argument("--bc_weights_path", type=str, default=None, help="Path to BC weights .pt file.")
parser.add_argument("--train_encoders", action="store_true", default=False, help="Enable end-to-end training of vision encoders (ResNet & PointNet).")
parser.add_argument("--resnet_weights_path", type=str, default=None, help="Path to ResNet-only weights .pt file.")
parser.add_argument("--pointnet_weights_path", type=str, default=None, help="Path to PointNet-only weights .pt file.")
parser.add_argument("--mlp_weights_path", type=str, default=None, help="Path to MLP Policy weights .pt file.")
parser.add_argument("--fresh_optimizer", action="store_true", default=False, help="Force rebuild optimizer (use for first train_encoders resume).")
parser.add_argument("--reset_noise_std", action="store_true", default=False, help="Clamp loaded policy noise std to 1.0 (keeps optimizer). Use when std has inflated during training.")
parser.add_argument("--override_lr", type=float, default=None, help="Force override learning rate when resuming.")
parser.add_argument("--override_std", type=float, default=None, help="Force override exploration std when resuming.")

# ------------------------------------------
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

class CustomOnPolicyRunner(OnPolicyRunner):
    """Custom runner to support mid-training weight resets without modifying the library."""
    def log(self, locs: dict, **kwargs):
        it = locs["it"]
        start_iter = locs["start_iter"] # To support relative calculation if needed
        
        # ========== Mid-Training Gripper Weight & Noise Reset ==========
        # Check if we should trigger a reset for re-exploration
        # 'gripper_reset_iteration' is in self.cfg which is a dict in OnPolicyRunner
        # [FIX] 暫時註解 Mid-Training Reset 邏輯，改用 VisionAsymActorCritic 內部的 std_min 控管
        # reset_it = self.cfg.get("gripper_reset_iteration", -1)
        # 
        # if it == reset_it:
        #     print(f"\n\033[1;33m[Custom Runner] Iteration {it}: Triggering gripper weight/bias reset for re-exploration.\033[0m")
        #     ac = self.alg.policy
        #     with torch.no_grad():
        #         # 1. Reset Noise Standard Deviation
        #         reset_std = self.cfg.get("gripper_reset_std", 1.0)
        #         if hasattr(ac, "std"):
        #             ac.std.fill_(reset_std)
        #         elif hasattr(ac, "log_std"):
        #             import math
        #             ac.log_std.fill_(math.log(reset_std))
        #         
        #         # 2. Reset Actor Last Layer (Gripper Dimension)
        #         actor_obj = getattr(ac, "actor", ac)
        #         linear_layers = [m for m in actor_obj.modules() if isinstance(m, torch.nn.Linear)]
        #         
        #         if linear_layers:
        #             last_linear = linear_layers[-1]
        #             if last_linear.out_features >= 1:
        #                 last_linear.weight[-1, :].fill_(0.0)
        #                 
        #                 # Set bias (default to 0.0, but user can set to -1.0 to force closing behavior initially)
        #                 reset_bias = self.cfg.get("gripper_reset_bias", 0.0)
        #                 last_linear.bias[-1].fill_(reset_bias)
        #                 print(f"[Custom Runner] Successfully reset weights/bias for {last_linear} (Bias={reset_bias})")
        
        # Call base logging
        super().log(locs, **kwargs)


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
try:
    import rsl_rl.runners.on_policy_runner
    import rsl_rl.runners.distillation_runner
    from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.agents import VisionAsymActorCritic
    # In rsl_rl, runners use eval() on the class_name.
    # We must inject the custom class into their global namespaces for eval() to find it.
    setattr(rsl_rl.runners.on_policy_runner, "VisionAsymActorCritic", VisionAsymActorCritic)
    setattr(rsl_rl.runners.distillation_runner, "VisionAsymActorCritic", VisionAsymActorCritic)
    print("[INFO] Registered VisionAsymActorCritic to rsl_rl runners.")
except ImportError:
    print("[WARN] Could not register VisionAsymActorCritic to all runners.")

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

    # --- Redesigned Implementation: Modular RL (Frozen or Trainable) ---
    # [FIX] 只要有 vision weights 或是開啟 train_encoders，就必須使用 VisionAsymActorCritic 
    # [FIX 2] 檢查 agent_cfg 本身是否就定義了要 policy_proprio (解決特定 agent_cfg 會報錯的問題)
    is_modular_vision = args_cli.train_encoders or args_cli.use_bc_weights or \
                        args_cli.resnet_weights_path or args_cli.pointnet_weights_path or \
                        "policy_proprio" in str(agent_cfg.obs_groups)
    
    print(f"[DEBUG] is_modular_vision: {is_modular_vision}")
    print(f"[DEBUG] agent_cfg.obs_groups: {agent_cfg.obs_groups}")
    
    if is_modular_vision:
        print(f"[INFO] Setting up Modular Vision Architecture (Train Encoders: {args_cli.train_encoders}).")
        # Load the fresh configuration for training from scratch
        from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.agents.rsl_rl_ppo_asym_fresh_refined_cfg import PPORunnerCfg as FreshPPORunnerCfg
        
        # Merge fresh config into existing agent_cfg (which was loaded by task registration)
        fresh_cfg = FreshPPORunnerCfg()
        agent_cfg.policy = fresh_cfg.policy
        agent_cfg.algorithm = fresh_cfg.algorithm
        agent_cfg.num_steps_per_env = fresh_cfg.num_steps_per_env
        agent_cfg.save_interval = fresh_cfg.save_interval
        
        # [CRITICAL] 必須開啟原始觀測，否則 rsl_rl 會因為找不到 policy_proprio 等 Key 而報錯
        env_cfg.use_raw_observations = True
        agent_cfg.policy.class_name = "VisionAsymActorCritic"
        
        if args_cli.use_bc_weights and args_cli.bc_weights_path:
            env_cfg.vision_weights_path = args_cli.bc_weights_path
            print(f"[INFO] Using BC weights for environment-side or initialization: {args_cli.bc_weights_path}")
    # ----------------------------------------------------------

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
    resume_path = None
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        # [FIX 0421] If load_checkpoint is an absolute path to a existing file, use it directly
        if agent_cfg.load_checkpoint and os.path.isfile(agent_cfg.load_checkpoint):
            resume_path = agent_cfg.load_checkpoint
        else:
            # Check if the log root path exists before scanning to avoid FileNotFoundError
            if not os.path.exists(log_root_path):
                # For distillation, we often start fresh but might need a teacher.
                # If we are NOT resuming a previous distillation run, but we are in Distillation mode,
                # we usually expect the user to provide a teacher checkpoint via absolute path.
                if agent_cfg.algorithm.class_name == "Distillation" and not agent_cfg.resume:
                    raise ValueError(
                        f"Log directory '{log_root_path}' does not exist for distillation experiment. "
                        "If you are training from scratch, please provide the teacher model path "
                        "using an absolute path in --checkpoint."
                    )
                else:
                    raise ValueError(f"Log directory '{log_root_path}' does not exist. Cannot resume.")
            
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
        # runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner = CustomOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # Get learning rates from config (used for modular setup or forced sync)
    base_lr = getattr(agent_cfg.algorithm, "learning_rate", 1e-4)
    encoder_lr = getattr(agent_cfg, "encoder_learning_rate", base_lr * 0.1)

    # ========== STEP 1: RE-INITIALIZE MODULAR OPTIMIZER (BEFORE model load) ==========
    if is_modular_vision:
        print(f"[INFO] Re-configuring optimizer for Modular Architecture (Train Encoders: {args_cli.train_encoders}).")
        policy = runner.alg.policy

        # Load explicit weights if provided
        if hasattr(policy, "load_modules"):
            policy.load_modules(
                resnet_path=args_cli.resnet_weights_path,
                pointnet_path=args_cli.pointnet_weights_path,
                mlp_path=args_cli.mlp_weights_path
            )

        params = []
        
        # 1. Handle Vision Encoders (Freeze if not --train_encoders)
        encoders = []
        if hasattr(policy, "resnet_encoder"): encoders.append(policy.resnet_encoder)
        if hasattr(policy, "pointnet_encoder"): encoders.append(policy.pointnet_encoder)
        
        if args_cli.train_encoders:
            print(f"[INFO] Vision Encoders: ENABLED for training (lr: {encoder_lr}).")
            if hasattr(policy, "resnet_encoder"):
                params.append({'params': policy.resnet_encoder.parameters(), 'lr': encoder_lr, 'name': 'resnet'})
            if hasattr(policy, "pointnet_encoder"):
                params.append({'params': policy.pointnet_encoder.parameters(), 'lr': encoder_lr, 'name': 'pointnet'})
        else:
            print("[INFO] Vision Encoders: FROZEN (eval mode, no gradients).")
            for enc in encoders:
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad = False

        # 2. Always Optimized: Actor/Critic MLP heads
        params.append({'params': policy.actor.parameters(), 'lr': base_lr, 'name': 'actor_mlp'})
        params.append({'params': policy.critic.parameters(), 'lr': base_lr, 'name': 'critic_mlp'})

        # 3. Always Optimized: Input LayerNorm
        if hasattr(policy, "actor_input_ln"):
            params.append({'params': policy.actor_input_ln.parameters(), 'lr': base_lr, 'name': 'actor_input_ln'})

        # 4. Always Optimized: Noise Parameters
        if hasattr(policy, "std"):
            params.append({'params': [policy.std], 'lr': base_lr, 'name': 'noise_std'})
        elif hasattr(policy, "log_std"):
            params.append({'params': [policy.log_std], 'lr': base_lr, 'name': 'noise_log_std'})

        # Initialize Optimizer
        from torch import optim
        runner.alg.optimizer = optim.Adam(params, lr=base_lr)

        print(f"[INFO] Respecting config schedule: {runner.alg.schedule} (desired_kl: {runner.alg.desired_kl})")

        # [VERIFY] Verify parameter groups
        for pg in runner.alg.optimizer.param_groups:
            print(f"  [Optimizer] Group '{pg.get('name', '?')}': lr={pg['lr']}, params={sum(p.numel() for p in pg['params'])}")

    # ========== STEP 2: LOAD CHECKPOINT ==========
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        
        # By default, retain Adam momentum unless specifically requested fresh
        load_opt = not args_cli.fresh_optimizer
        
        try:
            runner.load(resume_path, load_optimizer=load_opt)
            if load_opt:
                print(f"[INFO] Loaded model and optimizer from: {resume_path}")
            else:
                print(f"[INFO] Loaded model weights only (--fresh_optimizer). Target: {resume_path}")
            
            # [DEBUG]
            if hasattr(runner.alg.policy, "std"):
                print(f"[DEBUG 0401] policy.std.data after load: {runner.alg.policy.std.data}")
            elif hasattr(runner.alg.policy, "log_std"):
                print(f"[DEBUG 0401] policy.log_std.data after load: {runner.alg.policy.log_std.data}")

            # ========== STEP 2.1: FRESH OPTIMIZER & EXPLORATION RESET (0401) ==========
            if args_cli.fresh_optimizer:
                print(f"\033[1;32m[INFO] --fresh_optimizer detected. Resetting exploration noise and syncing LR.\033[0m")
                with torch.no_grad():
                    # 1. Reset Noise (std=1.0)
                    if hasattr(runner.alg.policy, "std"):
                        runner.alg.policy.std.data.fill_(1.0)
                        print(f"  [Reset] Forced policy.std = 1.0 (mean: {runner.alg.policy.std.mean().item():.2f})")
                    elif hasattr(runner.alg.policy, "log_std"):
                        runner.alg.policy.log_std.data.fill_(0.0)
                        print(f"  [Reset] Forced policy.log_std = 0.0 (std=1.0)")
                    
                    # 2. Reset Actor last layer (Gripper dim) if biased
                    # (Optional: break the learned 'stay open' bias)
                    # gripper_reset_bias = getattr(agent_cfg, "gripper_reset_bias", 0.0)
                    # print(f"  [Reset] Applying gripper reset bias: {gripper_reset_bias}")
                    # ... [Insert weight reset logic here if needed] ...

            # ========== STEP 2.2: NOISE STD RESET (WITHOUT OPTIMIZER DISCARD) ==========
            if args_cli.reset_noise_std:
                print(f"\033[1;33m[INFO] --reset_noise_std detected. Clamping policy noise std to max 1.0.\033[0m")
                with torch.no_grad():
                    if hasattr(runner.alg.policy, "std"):
                        before = runner.alg.policy.std.data.clone()
                        runner.alg.policy.std.data.clamp_(max=1.0)
                        after = runner.alg.policy.std.data
                        print(f"  [Reset] policy.std clamped: {before.mean().item():.3f} → {after.mean().item():.3f}")
                    elif hasattr(runner.alg.policy, "log_std"):
                        runner.alg.policy.log_std.data.clamp_(max=0.0)  # log(1.0) = 0.0
                        print(f"  [Reset] policy.log_std clamped to max 0.0 (std=1.0)")

            # ========== STEP 2.3: MANUAL OVERRIDES (LR & STD) ==========
            if args_cli.override_lr is not None:
                print(f"\033[1;32m[INFO] --override_lr detected. Setting LR to {args_cli.override_lr}\033[0m")
                runner.alg.learning_rate = args_cli.override_lr
                for i, param_group in enumerate(runner.alg.optimizer.param_groups):
                    param_group['lr'] = args_cli.override_lr
                    print(f"  [Optimizer Sync] Group {i} ('{param_group.get('name', '?')}'): lr set to {param_group['lr']}")

            if args_cli.override_std is not None:
                print(f"\033[1;32m[INFO] --override_std detected. Setting policy noise std to {args_cli.override_std}\033[0m")
                with torch.no_grad():
                    if hasattr(runner.alg.policy, "std"):
                        runner.alg.policy.std.data.fill_(args_cli.override_std)
                    elif hasattr(runner.alg.policy, "log_std"):
                        import math
                        runner.alg.policy.log_std.data.fill_(math.log(args_cli.override_std))

            # This ensures that even if the checkpoint had 1e-5, we start at base_lr (1e-4)
            # [RESTORE] 僅在 modular_vision 模式下強制同步 LR，舊版恢復官方預設模式以利恢復 adaptive lr
            if is_modular_vision and args_cli.override_lr is None:
                target_lr = base_lr
                runner.alg.learning_rate = target_lr
                for i, param_group in enumerate(runner.alg.optimizer.param_groups):
                    # If vision encoders are optimized, they might use a different LR (encoder_lr)
                    if 'name' in param_group and (param_group['name'] == 'resnet' or param_group['name'] == 'pointnet'):
                        param_group['lr'] = encoder_lr
                    else:
                        param_group['lr'] = target_lr
                    print(f"  [Optimizer Sync] Group {i} ('{param_group.get('name', '?')}'): lr set to {param_group['lr']}")

        except Exception as e:
            # Fallback for strict group mismatch
            print(f"\033[1;33m[WARN] Optimizer load failed ({e}). Falling back to load weights only.\033[0m")
            runner.load(resume_path, load_optimizer=False)
            print(f"[INFO] Loaded model weights only. Target: {resume_path}")
            
            # Sync LR even on fallback (only if modular_vision)
            if is_modular_vision:
                runner.alg.learning_rate = base_lr
                for param_group in runner.alg.optimizer.param_groups:
                    param_group['lr'] = base_lr


    # ========== STEP 3: SYNC BARRIER BEFORE TRAINING (HIGH-RELIABILITY RESUME) ==========
    if args_cli.distributed:
        import torch.distributed as dist
        if dist.is_initialized():
            print(f"\033[1;34m[INFO] Waiting for all ranks to complete checkpoint loading (Rank {dist.get_rank()})...\033[0m")
            dist.barrier()
            print(f"\033[1;32m[INFO] Synchronization complete. Entering training loop (Rank {dist.get_rank()}).\033[0m")
    # ====================================================================================

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
