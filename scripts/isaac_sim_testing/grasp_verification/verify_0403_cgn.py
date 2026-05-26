# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verification script for 0403 CGN Debug Features."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Verify 0403 CGN Debug.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
import os
import shutil

# local imports
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0403_env_cfg import PickupPlaceVisionAsym0403EnvCfg
import pickup_place_direct_0203.tasks # registers tasks

def main():
    # 1. Prepare Config
    env_cfg = PickupPlaceVisionAsym0403EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Enable Debug Features
    env_cfg.cgn_debug_vis = True
    env_cfg.cgn_debug_snapshots = True
    env_cfg.cgn_debug_dir = "logs/verify_cgn_0403"
    
    # [0406 VERIFY] Force zero threshold to ensure we get a comparison image
    env_cfg.cgn_score_threshold = 0.0
    env_cfg.cgn_proximity_filter = 1.5 # meters
    env_cfg.cgn_top_k = 10
    
    # Clear previous logs
    if os.path.exists(env_cfg.cgn_debug_dir):
        shutil.rmtree(env_cfg.cgn_debug_dir)
    
    # 2. Create Env
    print("[VERIFY] Creating env: Pickup-Place-Direct-Vision-Asym-v3")
    env = gym.make("Pickup-Place-Direct-Vision-Asym-v3", cfg=env_cfg)
    
    # 3. Run Simulation
    print("[VERIFY] Running 3 episodes (25 steps each)...")
    
    actual_env = env.unwrapped
    action_dim = actual_env.action_space.shape[0] if hasattr(actual_env.action_space, 'shape') else 6
    device = actual_env.device
    num_envs = actual_env.num_envs

    for ep in range(3):
        print(f"\n[VERIFY] Episode {ep} - Resetting...")
        obs, _ = env.reset()
        
        # [0404 Fix] Sync camera
        actual_env.sim.render()
        
        print(f"[VERIFY] Env initialized. num_envs={num_envs}, action_dim={action_dim}, device={device}")

        for i in range(25):
            # Random actions
            actions = 2.0 * torch.rand((num_envs, action_dim), device=device) - 1.0
            obs, rewards, terminations, truncations, extras = env.step(actions)
            
            if i % 10 == 0:
                dist = extras.get("cgn_dist_m", "N/A")
                align = extras.get("cgn_align_deg", "N/A")
                print(f"  Step {i}: Grasp Gap dist={dist}, align={align}")

    # 4. Check for Artifacts
    if os.path.exists(env_cfg.cgn_debug_dir):
        files = os.listdir(env_cfg.cgn_debug_dir)
        print(f"[VERIFY] Found {len(files)} debug files in {env_cfg.cgn_debug_dir}")
        for f in files:
            print(f"  - {f}")
    else:
        print("[VERIFY] ERROR: Debug directory not created!")

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
