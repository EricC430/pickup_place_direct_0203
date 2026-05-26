import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from collections import deque
from pathlib import Path

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Setup App Launcher before other imports! Critical for Omniverse initialization.
parser = argparse.ArgumentParser(description="Online Behavioral Cloning")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=100, help="Maximum number of BC iterations.")
parser.add_argument("--run_name", type=str, default="default_run", help="Name of the run for organizing checkpoints.")
parser.add_argument("--save_interval", type=int, default=500, help="Interval in iterations to save the model.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval (in steps) between recording videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

# Environment imports
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.mdp import observations as mdp_obs
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0310_env_cfg import PickupPlaceVisionAsym0310EnvCfg
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0310_env import PickupPlaceVisionAsym0310Env


class StudentActor(nn.Module):
    def __init__(self, vision_encoder_low, pointnet, input_dim=1130, action_dim=6):
        super().__init__()
        # Embed the vision encoders into the model so they get saved!
        self.vision_encoder_low = copy.deepcopy(vision_encoder_low)
        self.pointnet = copy.deepcopy(pointnet)
        
        # Vision projection layer: project 1088 vision dims down to 64
        self.vision_proj = nn.Sequential(
            nn.Linear(1088, 64),
            nn.ELU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(42 + 64, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, policy_obs, rgb_raw, ptcloud_world):
        # 1. Preprocess RGB (match Isaac environment logic)
        rgb_norm = rgb_raw[..., :3] / 255.0
        rgb_norm = torch.clamp(rgb_norm, 0.0, 1.0)
        rgb_chw = rgb_norm.permute(0, 3, 1, 2).float()
        
        # 2. Forward CNN & PointNet (Tracking Gradients!)
        vision_low_current = self.vision_encoder_low(rgb_chw)
        # PointNet expects (B, N, 3) where N=1024
        pointnet_current = self.pointnet(ptcloud_world)
        
        # 3. Splice out the components from the environment's observation
        # JPos(6) + JVel(6) + JErr(6) + Last4Actions(24) = 42
        proprio = policy_obs[:, :42]
        vision_low_history = policy_obs[:, 42:554].clone()  # 512 dims
        pointnet_history = policy_obs[:, 554:1066].clone()  # 512 dims
        vision_high = policy_obs[:, 1066:]                  # 64 dims
        
        # 4. Replace the LAST frame in histories with our fresh gradient-tracked features
        vision_low_history[:, -128:] = vision_low_current
        pointnet_history[:, -128:] = pointnet_current
        
        # 5. Project vision features
        vision_features = torch.cat([vision_low_history, pointnet_history, vision_high], dim=-1)
        vision_embed = self.vision_proj(vision_features)
        
        # 6. Recombine and pass to MLP
        x = torch.cat([proprio, vision_embed], dim=-1)
        
        # Directly predict normalized absolute actions to match Teacher
        predicted_actions = self.mlp(x)
        
        return predicted_actions

class StudentCritic(nn.Module):
    def __init__(self, input_dim=73):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.mlp(x)

def get_teacher_obs(env):
    """
    Extracts 46-dimensional privileged observation for the teacher model.
    Structure: [jpos_arm(5), jvel_arm(5), obj_pos(3), bbox(24), target(3), actions(6)]
    """
    ue = env.unwrapped
    
    jpos_arm = ue.joint_pos[:, ue._arm_joint_indices] - ue.robot.data.default_joint_pos[:, ue._arm_joint_indices] # (B, 5)
    jvel_arm = ue.joint_vel[:, ue._arm_joint_indices] - ue.robot.data.default_joint_vel[:, ue._arm_joint_indices] # (B, 5)
    
    obj_pos = mdp_obs.object_position_in_robot_root_frame(ue, object_cfg=SceneEntityCfg("object"))
    bbox = mdp_obs.object_bbox_corners_relative(ue, object_cfg=SceneEntityCfg("object"))
    
    target_pos_w = ue.target_poses + ue.scene.env_origins
    target, _ = subtract_frame_transforms(
        ue.scene["robot"].data.root_pos_w, 
        ue.scene["robot"].data.root_quat_w, 
        target_pos_w
    )
    
    teacher_obs = torch.cat([jpos_arm, jvel_arm, obj_pos, bbox, target, ue.actions], dim=-1) # (B, 46)
    return teacher_obs

def main():
    env_cfg = PickupPlaceVisionAsym0310EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Initialize the environment with rendering if video is requested
    render_mode = "rgb_array" if args_cli.video else None
    env = PickupPlaceVisionAsym0310Env(cfg=env_cfg, render_mode=render_mode)
    
    # Wrap with video recorder if requested
    if args_cli.video:
        import gymnasium as gym
        video_dir = os.path.join("logs", "bc_runs", args_cli.run_name, "videos")
        os.makedirs(video_dir, exist_ok=True)
        print(f"[INFO] Video recording enabled! Saving to: {video_dir}")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True
        )
        
    device = env.unwrapped.device
    
    teacher_path = "/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/proprioception_only/2026-02-12_00-12-19/exported/policy.pt"
    if os.path.exists(teacher_path):
        teacher_actor = torch.jit.load(teacher_path).to(device)
    else:
        print(f"Teacher model not found at {teacher_path}. Creating mock teacher for testing syntax.")
        teacher_actor = nn.Sequential(
            nn.Linear(46, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 6)
        ).to(device)

    teacher_actor.eval()
    for param in teacher_actor.parameters():
        param.requires_grad = False
        
    obs, _ = env.reset()
    student_dim = obs["policy"].shape[-1]
    critic_dim = obs["critic"].shape[-1]
    
    # Pass the environment's vision encoders into the StudentActor so it gets its own copies
    student_actor = StudentActor(
        vision_encoder_low=env.unwrapped.vision_encoder_low,
        pointnet=env.unwrapped.pointnet,
        input_dim=student_dim
    ).to(device)
    
    student_critic = StudentCritic(input_dim=critic_dim).to(device)
    
    student_actor.train()
    
    # Configure initial optimizer: Only train Student Actor MLP
    optimizer = optim.Adam(student_actor.mlp.parameters(), lr=1e-4)
    
    # Freeze ResNet and PointNet initially
    for param in student_actor.vision_encoder_low.parameters():
        param.requires_grad = False
    for param in student_actor.pointnet.parameters():
        param.requires_grad = False
            
    loss_fn = nn.L1Loss()
    
    privileged_obs = get_teacher_obs(env)
    print(f"Teacher Obs Dim: {privileged_obs.shape[-1]}")
    print(f"Student Obs Dim: {student_dim}")
    print(f"Critic Obs Dim:  {critic_dim}")
    print("Starting Training Loop...")
    
    # Define when to unfreeze ResNet (e.g., halfway through training)
    unfreeze_iteration = args_cli.max_iterations // 2
    
    for iteration in range(args_cli.max_iterations):
        # ---------------- ResNet Unfreezing Logic ----------------
        if iteration == unfreeze_iteration:
            print(f"Iteration {iteration}: Unfreezing ResNet for fine-tuning with very small LR.")
            # Unfreeze the vision parameters
            for param in student_actor.vision_encoder_low.parameters():
                param.requires_grad = True
            for param in student_actor.pointnet.parameters():
                param.requires_grad = True
            
            # Add to optimizer with a very small learning rate
            optimizer.add_param_group({'params': student_actor.vision_encoder_low.parameters(), 'lr': 1e-6})
            optimizer.add_param_group({'params': student_actor.pointnet.parameters(), 'lr': 1e-6})
        # ---------------------------------------------------------

        privileged_obs = get_teacher_obs(env)
        with torch.no_grad():
            teacher_actions = teacher_actor(privileged_obs)
        
        policy_obs = obs["policy"]
        
        # Extract raw images and pre-computed pointcloud from the environment for End-to-End processing
        rgb_raw = env.unwrapped.camera_low.data.output["rgb"]
        ptcloud_world = env.unwrapped.current_ptcloud
        
        # Forward pass tracking gradients all the way back to the ResNet and PointNet for the current frame
        student_actions = student_actor(policy_obs, rgb_raw, ptcloud_world)
        
        # RL wrappers (like rsl_rl) automatically clamp the raw continuous Actor outputs to [-1.0, 1.0] 
        # before passing them to the physics engine. We MUST use this clamped version as our BC regression target!
        clamped_teacher_actions = torch.clamp(teacher_actions, -1.0, 1.0)
        
        # Supervised BC Loss
        loss = loss_fn(student_actions, clamped_teacher_actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ---------------------------------------------------------
        # Step environment using the expert's CLAMPED action to keep the trajectory stable and successful
        obs, reward, terminated, truncated, extras = env.step(clamped_teacher_actions)
        # ---------------------------------------------------------

        # --- Analytics and Logging ---
        # Calculate L1 error (absolute difference) to understand specifically where it fails
        with torch.no_grad():
            l1_error = torch.abs(student_actions - clamped_teacher_actions).mean(dim=0) # Shape: (6,)
            arm_error = l1_error[:5].mean().item()
            gripper_error = l1_error[5].item()
            
            print(f"Iteration [{iteration+1}/{args_cli.max_iterations}] - Loss: {loss.item():.6f} | L1 Arm: {arm_error:.4f} | L1 Gripper: {gripper_error:.4f}")
            
            # Print the first environment's actual actions periodically to visually verify
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  Env 0 T-Action (Clamped): {clamped_teacher_actions[0].cpu().numpy().round(3)}")
                print(f"  Env 0 S-Action: {student_actions[0].detach().cpu().numpy().round(3)}")
                
        # --- Checkpointing Logic ---
        if (iteration + 1) % args_cli.save_interval == 0:
            save_dir = os.path.join("logs", "bc_runs", args_cli.run_name)
            os.makedirs(save_dir, exist_ok=True)
            model_name = f"model_iter_{iteration+1}.pt"
            save_path = os.path.join(save_dir, model_name)
            torch.save(student_actor.state_dict(), save_path)
            print(f"[INFO] Saved Checkpoint: {save_path}")
            
    # Final save after loops
    print(f"Finished Training! Saving final model...")
    final_dir = os.path.join("logs", "bc_runs", args_cli.run_name)
    os.makedirs(final_dir, exist_ok=True)
    final_path = os.path.join(final_dir, "model_final.pt")
    torch.save(student_actor.state_dict(), final_path)
    print(f"[INFO] Final StudentActor state_dict saved to {final_path}")

if __name__ == "__main__":
    main()
