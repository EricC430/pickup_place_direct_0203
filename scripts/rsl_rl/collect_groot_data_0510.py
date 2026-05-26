# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint and collect successful trajectories for GR00T.
   Adapted for the 0510 Task-Space Delta IK environment.
"""

import argparse
import sys
import os
import torch
import numpy as np
import json
import csv
import shutil
from PIL import Image

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect GR00T trajectory data (0510 Delta IK).")
parser.add_argument("--num_episodes", type=int, default=4, help="Number of successful episodes to collect.")
parser.add_argument("--task", type=str, default="Template-Pickup-Place-Direct-0510-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--object_id", type=int, required=True, help="The specific object ID to spawn.")
parser.add_argument("--output_dir", type=str, default="groot_data_raw", help="Directory to save collected data.")
parser.add_argument("--task_description", type=str, required=True, help="Language description of the task.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Force PyTorch CUDA context initialization
try:
    if torch.cuda.is_available():
        _ = torch.zeros(1, device="cuda")
except Exception:
    pass

# Force enable cameras for rendering wrist camera
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, DirectMARLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ensure local task is in path
ext_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "pickup_place_direct_0203"))
if ext_dir not in sys.path:
    sys.path.append(ext_dir)

import pickup_place_direct_0203.tasks  # noqa: F401
from isaaclab.utils.math import subtract_frame_transforms


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Ensure 1 environment for isolated recording
    env_cfg.scene.num_envs = 1
    
    # 0510 decimation=10, dt=0.01 -> step_dt=0.1s, episode=5s -> 50 steps per episode
    max_steps = 50
    WARMUP_STEPS = 5
    
    # Configure wrist camera (matches Orbbec Dabai DCW specs)
    wrist_camera_cfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/depth_cam_link/camera_mount_marker/Camera_GR00T",
        update_period=0.0,                          # Sync with sim step
        height=480,
        width=640,
        data_types=["rgb"],                         # RGB only
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.7,                      # HFOV = 79°
            horizontal_aperture=20.955,
            clipping_range=(0.01, 3.8),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),            # ROS optical convention
            convention="ros",
        ),
    )
    env_cfg.wrist_camera_cfg = wrist_camera_cfg

    # Configure environment to spawn ONLY specified object ID
    obj_id = args_cli.object_id
    print(f"\n[INFO] Configuring environment to spawn object ID: {obj_id}")
    
    if hasattr(env_cfg, "object_cfg"):
        env_cfg.object_cfg.spawn.assets_cfg = [
            sim_utils.UsdFileCfg(
                usd_path=f"/workspace/test_isaaclab/ObjectFolder_selected/{obj_id}/{obj_id}.usd",
                scale=(0.6, 0.6, 0.6),
            )
        ]
        
    # Modify SELECTED_OBJECT_IDS in-place across config modules
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod_name.endswith("pickup_place_direct_0510_env_cfg") and mod is not None:
            if hasattr(mod, "SELECTED_OBJECT_IDS"):
                mod.SELECTED_OBJECT_IDS.clear()
                mod.SELECTED_OBJECT_IDS.append(obj_id)
    try:
        import pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_direct_0510_env_cfg as env_cfg_mod
        env_cfg_mod.SELECTED_OBJECT_IDS.clear()
        env_cfg_mod.SELECTED_OBJECT_IDS.append(obj_id)
    except Exception as e:
        print(f"[WARNING] Failed to update SELECTED_OBJECT_IDS: {e}")
    
    # 0510 uses 49D observation space
    if hasattr(env_cfg, "use_46_dim_obs"):
        env_cfg.use_46_dim_obs = False
    env_cfg.observation_space = 49
    env_cfg.wait_for_textures = False

    # Override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Adjust camera position for spectator viewer
    env_cfg.viewer.eye = [1.0, -1.0, 1.0]
    env_cfg.viewer.lookat = [0.15, 0.0, 0.1]
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Wrap for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    # Load weights
    try:
        policy_module = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        loaded_dict = torch.load(resume_path, map_location=agent_cfg.device)
        if "model_state_dict" in loaded_dict:
            policy_module.load_state_dict(loaded_dict["model_state_dict"])
        else:
            policy_module.actor.load_state_dict(loaded_dict, strict=False)
    except Exception:
        runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Get joint indices
    arm_joint_indices = env.unwrapped._arm_joint_indices
    gripper_joint_idx = env.unwrapped._gripper_joint_idx
    controlled_joint_ids = list(arm_joint_indices) + list(gripper_joint_idx)

    # Output directories setup
    raw_out_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(raw_out_dir, exist_ok=True)
    print(f"[INFO] Saving collected trajectories to: {raw_out_dir}")

    success_episodes_saved = 0
    total_episodes_attempted = 0

    while success_episodes_saved < args_cli.num_episodes:
        total_episodes_attempted += 1
        print(f"\n--- Attempting Episode (Attempt {total_episodes_attempted}, Saved {success_episodes_saved}/{args_cli.num_episodes}) ---")
        
        # Temp buffers for the current episode
        temp_frames = []
        temp_data = []
        
        # Reset environment and step loop under inference mode
        extras = {}
        with torch.inference_mode():
            obs, _ = env.reset()
            
            step_count = 0
            warmup_joint_pos = None
            
            while step_count < max_steps:
                # Warmup interception
                if step_count < WARMUP_STEPS:
                    if warmup_joint_pos is None:
                        warmup_joint_pos = env.unwrapped.scene["robot"].data.joint_pos.clone()
                    
                    env.unwrapped.scene["robot"].set_joint_position_target(warmup_joint_pos)
                    zero_actions = torch.zeros_like(policy(obs))
                    obs, _, dones, extras = env.step(zero_actions)
                    step_count += 1
                    continue
                
                # Active policy stepping
                actions = policy(obs)
                
                # Retrieve camera frame (TiledCamera of wrist)
                # output['rgb'] has shape (1, 480, 640, 4) or similar
                rgb_tensor = env.unwrapped.wrist_camera.data.output["rgb"]
                rgb_np = rgb_tensor[0, :, :, :3].cpu().numpy().astype(np.uint8)
                
                # Retrieve actual joint pos (6D) and target joint pos (6D)
                actual_joints = env.unwrapped.scene["robot"].data.joint_pos[:, controlled_joint_ids].cpu().numpy().flatten()
                target_joints = env.unwrapped.scene["robot"].data.joint_pos_target[:, controlled_joint_ids].cpu().numpy().flatten()
                
                # Retrieve Goal relative to gripper and relative to object in world/env frame
                target_pos_w = env.unwrapped.target_poses + env.unwrapped.scene.env_origins
                ee_pos_w = env.unwrapped.ee_frame.data.target_pos_w[:, 0, :]
                object_pos_w = env.unwrapped.object.data.root_pos_w
                
                goal_rel_ee = (target_pos_w - ee_pos_w).cpu().numpy().flatten()
                goal_rel_obj = (target_pos_w - object_pos_w).cpu().numpy().flatten()
                
                # Save step data
                temp_frames.append(rgb_np)
                temp_data.append({
                    "step": step_count - WARMUP_STEPS, # 0-indexed for active steps
                    "joint_pos": actual_joints.tolist(),
                    "joint_target": target_joints.tolist(),
                    "goal_rel_ee": goal_rel_ee.tolist(),
                    "goal_rel_obj": goal_rel_obj.tolist()
                })
                
                # Step environment
                obs, _, dones, extras = env.step(actions)
                
                if dones.any():
                    break
                
                step_count += 1
        
        # Determine if episode was successful
        # (lifting_success and object_goal_tracking_success both true)
        is_success = False
        if "episode" in extras:
            ep_info = extras["episode"]
            lifting = ep_info.get("lifting_success", torch.zeros(1)).any().item()
            goal_tracking = ep_info.get("object_goal_tracking_success", torch.zeros(1)).any().item()
            is_success = bool(lifting and goal_tracking)
            print(f"Episode completed. Lifting: {lifting}, Goal Tracking: {goal_tracking} -> Success: {is_success}")
        else:
            print("Episode completed, but no episode metrics found in extras.")
            
        if is_success:
            # Save the successful episode
            ep_idx = success_episodes_saved
            ep_dir = os.path.join(raw_out_dir, f"episode_{ep_idx:06d}")
            img_dir = os.path.join(ep_dir, "images")
            os.makedirs(img_dir, exist_ok=True)
            
            # Save frames
            for idx, frame in enumerate(temp_frames):
                img = Image.fromarray(frame)
                img.save(os.path.join(img_dir, f"frame_{idx:06d}.png"))
                
            # Save CSV file
            csv_path = os.path.join(ep_dir, "data.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                # Header
                header = ["step"]
                for j in range(6): header.append(f"joint_pos_{j}")
                for j in range(6): header.append(f"joint_target_{j}")
                for j in range(3): header.append(f"goal_rel_ee_{j}")
                for j in range(3): header.append(f"goal_rel_obj_{j}")
                writer.writerow(header)
                
                # Rows
                for row_data in temp_data:
                    row = ([row_data["step"]] + 
                           row_data["joint_pos"] + 
                           row_data["joint_target"] + 
                           row_data["goal_rel_ee"] + 
                           row_data["goal_rel_obj"])
                    writer.writerow(row)
            
            # Save Metadata JSON
            meta = {
                "episode_idx": ep_idx,
                "object_id": obj_id,
                "task_description": args_cli.task_description,
                "success": True,
                "num_steps": len(temp_data)
            }
            with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=4)
                
            print(f"Saved successful episode {ep_idx} to {ep_dir}")
            success_episodes_saved += 1
        else:
            print("Discarded unsuccessful episode.")

    # Close env
    env.close()
    print(f"\n[SUCCESS] Successfully collected {args_cli.num_episodes} trajectories for object ID {obj_id}!")


if __name__ == "__main__":
    main()
    simulation_app.close()
