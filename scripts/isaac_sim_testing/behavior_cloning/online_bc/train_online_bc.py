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


class EmpiricalNormalizer(nn.Module):
    """
    Normalizes observations using running mean and variance.
    Matches the RSL_RL 'EmpiricalNormalization' behavior.
    Now follows standard nn.Module device handling (moving via .to(device)).
    """
    def __init__(self, shape):
        super().__init__()
        # Use register_buffer without explicit device; it moves with the module
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1.0))
        self.epsilon = 1e-8

    def forward(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def update(self, x):
        """Update running stats based on a batch of data x."""
        with torch.no_grad():
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)
            batch_count = x.shape[0]

            delta = batch_mean - self.running_mean
            new_count = self.count + batch_count

            self.running_mean += delta * batch_count / new_count
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + delta**2 * self.count * batch_count / new_count
            self.running_var = m_2 / new_count
            self.count = new_count

    def load_states(self, state_dict):
        # rsl_rl compat: handle dicts from either policy.pt or model.pt
        self.load_state_dict(state_dict, strict=False)


class StudentActor(nn.Module):
    def __init__(self, vision_encoder_low, pointnet, input_dim=1130, action_dim=6):
        super().__init__()
        # Embed the vision encoders into the model so they get saved!
        self.vision_encoder_low = copy.deepcopy(vision_encoder_low)
        self.pointnet = copy.deepcopy(pointnet)
        
        # 1. Proprio Normalization: Use EmpiricalNormalizer per recommendation
        self.proprio_norm = EmpiricalNormalizer(42)
        
        # 2. Vision Normalization: Split into separate LayerNorms to prevent 'feature bullying'
        # CNN Low-Res History (512), PointNet History (512), CNN High-Res Single (64)
        self.vision_low_ln = nn.LayerNorm(512)
        self.pointnet_ln = nn.LayerNorm(512)
        self.vision_high_ln = nn.LayerNorm(64)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim)
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
        
        # 3. Splice out components and split into Proprio vs Vision
        # Proprio block: JPos(6) + JVel(6) + JErr(6) + Last4Actions(24) = 42
        proprio_obs = policy_obs[:, :42]
        # Update proprio stats during training
        if self.training:
            self.proprio_norm.update(proprio_obs)
        proprio_normed = self.proprio_norm(proprio_obs)
        
        # Vision blocks
        vision_low_history = policy_obs[:, 42:554].clone()  # 512 dims
        pointnet_history = policy_obs[:, 554:1066].clone()  # 512 dims
        vision_high = policy_obs[:, 1066:]                  # 64 dims
        
        # 4. Replace the LAST frame in histories with our fresh gradient-tracked features
        vision_low_history[:, -128:] = vision_low_current
        pointnet_history[:, -128:] = pointnet_current
        
        # 5. Apply separate LayerNorms to prevent feature dominance imbalances
        vision_low_normed = self.vision_low_ln(vision_low_history)
        pointnet_normed = self.pointnet_ln(pointnet_history)
        vision_high_normed = self.vision_high_ln(vision_high)
        
        # 6. Recombine and pass to MLP
        x = torch.cat([proprio_normed, vision_low_normed, pointnet_normed, vision_high_normed], dim=-1)
        # return torch.tanh(self.mlp(x)) # RE-ENABLE TANH IF TRAINING STILL STAGNATES
        return self.mlp(x)

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
    
    IMPORTANT: We use fixed Teacher Default Joints to ensure the spatial reference 
    matches original training, even if the user changed the environment defaults.
    """
    ue = env.unwrapped
    
    # These MUST match the teacher's training environment defaults exactly
    TEACHER_DEFAULT_JPOS = torch.tensor([0.0, 0.61086472, 0.7853975, 0.95993027, 0.0], device=ue.device)
    TEACHER_DEFAULT_JVEL = torch.zeros(5, device=ue.device)
    
    # Current joint positions and velocities for arm
    jpos_arm = ue.joint_pos[:, ue._arm_joint_indices] - TEACHER_DEFAULT_JPOS
    jvel_arm = ue.joint_vel[:, ue._arm_joint_indices] - TEACHER_DEFAULT_JVEL
    
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
    
    # Identify the environment closest to the world center [0, 0, 0] for better visualization
    env_origins = env.unwrapped.scene.env_origins # (num_envs, 3)
    distances = torch.norm(env_origins, dim=-1)
    target_env_idx = torch.argmin(distances).item()
    print(f"[INFO] POV Video target environment: {target_env_idx} (closest to world origin)")
    
    teacher_path = "/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/proprioception_only/2026-02-12_00-12-19/exported/policy.pt" #"/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/proprioception_only/2026-02-12_00-12-19/model_2700.pt" #  # "/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/manager_to_direct_test/2026-02-08_03-13-49/model_5700.pt" 
    if os.path.exists(teacher_path):
        try:
            # 1. Try JIT loading (for exported/policy.pt)
            teacher_actor = torch.jit.load(teacher_path).to(device)
            teacher_actor.is_jit = True  # Custom flag
            print(f"[INFO] Loaded JIT teacher model from {teacher_path}")
        except Exception as e:
            # 2. Fallback for regular checkpoints (model_XXXX.pt)
            print(f"[INFO] JIT load failed, attempting to load as state_dict checkpoint...")
            teacher_actor = nn.Sequential(
                nn.Linear(46, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, 64), nn.ELU(),
                nn.Linear(64, 6)
            ).to(device)
            teacher_actor.is_jit = False  # Custom flag
            
            ckpt = torch.load(teacher_path, map_location=device)
            raw_state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            
            # Map keys: rsl_rl usually uses "actor.0.weight", we need "0.weight"
            # And we need to account for Sequential indices (0, 2, 4, 6 for Linear layers)
            new_state_dict = {}
            for k, v in raw_state_dict.items():
                if k.startswith("actor."):
                    new_state_dict[k.replace("actor.", "")] = v
            
            try:
                teacher_actor.load_state_dict(new_state_dict, strict=False)
                print(f"[INFO] Successfully loaded state_dict into teacher MLP.")
            except Exception as e2:
                print(f"[ERROR] Failed to load state_dict: {e2}")
                
            # 3. Load Normalization stats
            teacher_normalizer = EmpiricalNormalizer(shape=(46,))
            teacher_normalizer.to(device)
            try:
                # Normalizer stats are usually in ckpt['obs_normalizer']
                if "obs_normalizer" in ckpt:
                    teacher_normalizer.load_states(ckpt["obs_normalizer"]["policy"])
                    print(f"[INFO] Loaded Observation Normalizer stats.")
                else:
                    print(f"[WARNING] No normalization stats found in checkpoint. Policy might be inaccurate.")
            except Exception as e3:
                print(f"[WARNING] Failed to load normalization stats: {e3}")
    else:
        print(f"Teacher model not found at {teacher_path}. Creating mock teacher for testing syntax.")
        teacher_actor = nn.Sequential(
            nn.Linear(46, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, 6)
        ).to(device)
        teacher_actor.is_jit = False

    teacher_actor.eval()
    if 'teacher_normalizer' not in locals():
        # Fallback if no normalizer was loaded (e.g. mock teacher)
        teacher_normalizer = nn.Identity().to(device)
        
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
            
    loss_fn = nn.SmoothL1Loss(beta=0.05) # Smaller beta for finer precision near zero
    
    privileged_obs = get_teacher_obs(env)
    print(f"Teacher Obs Dim: {privileged_obs.shape[-1]}")
    print(f"Student Obs Dim: {student_dim}")
    print(f"Critic Obs Dim:  {critic_dim}")
    print("Starting Training Loop...")
    
    # ---------------- Video Recording (On-Robot POV) ----------------
    on_robot_video_frames = []
    import torchvision
    
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
            
            # Add to optimizer with a slightly larger, yet safe, learning rate
            # 1e-6 was likely way too small for vision encoders to learn anything within 40k iters.
            optimizer.add_param_group({'params': student_actor.vision_encoder_low.parameters(), 'lr': 2e-5})
            optimizer.add_param_group({'params': student_actor.pointnet.parameters(), 'lr': 2e-5})
        # ---------------------------------------------------------

        privileged_obs = get_teacher_obs(env)
        with torch.no_grad():
            # If load from state_dict, we MUST normalize manually.
            # If JIT (policy.pt), RSL_RL typically bundles Normalizer INSIDE the JIT.
            # However, to be safe and follow user instruction: we apply teacher_normalizer 
            # ONLY if the actor doesn't already have it or if it's a fallback MLP.
            if getattr(teacher_actor, 'is_jit', False):
                # Check if the JIT model actually contains normalization. 
                # Most standard RSL_RL policy.pt DO. If so, we skip manual norm to avoid double-scaling.
                # If you are CERTAIN your policy.pt is raw MLP, change this logic.
                teacher_actions = teacher_actor(privileged_obs)
            else:
                norm_privileged_obs = teacher_normalizer(privileged_obs)
                teacher_actions = teacher_actor(norm_privileged_obs)

        # RL wrappers (like rsl_rl) automatically clamp the raw continuous Actor outputs to [-1.0, 1.0] 
        # before passing them to the physics engine. Since we are bypassing the wrapper, we must clamp them manually!
        clamped_teacher_actions = torch.clamp(teacher_actions, -1.0, 1.0)
        
        policy_obs = obs["policy"]
        
        # Extract raw images and pre-computed pointcloud from the environment for End-to-End processing
        rgb_raw = env.unwrapped.camera_low.data.output["rgb"]
        ptcloud_world = env.unwrapped.current_ptcloud
        
        # Forward pass tracking gradients all the way back to the ResNet and PointNet for the current frame
        student_actions = student_actor(policy_obs, rgb_raw, ptcloud_world)

        # Supervised BC Loss
        loss = loss_fn(student_actions, clamped_teacher_actions)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ---------------------------------------------------------
        # # Step environment using the expert's action to gather the next state
        # obs, reward, terminated, truncated, extras = env.step(teacher_actions)
        # --- Physics Execution ---
        
        # Step environment using the expert's CLAMPED action to keep the trajectory stable and successful
        obs, reward, terminated, truncated, extras = env.step(clamped_teacher_actions)
        # ---------------------------------------------------------

        # --- Analytics and Logging ---
        # Calculate L1 error (absolute difference) to understand specifically where it fails
        with torch.no_grad():
            l1_error = torch.abs(student_actions - clamped_teacher_actions).mean(dim=0) # Shape: (6,)
            arm_error = l1_error[:5].mean().item()
            gripper_error = l1_error[5].item()
            
            print(f"Iteration [{iteration+1}/{args_cli.max_iterations}] - SmoothL1 Loss: {loss.item():.6f} | L1 Arm: {arm_error:.4f} | L1 Gripper: {gripper_error:.4f}")
            
            # Print the first environment's actual actions periodically to visually verify
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  Env {target_env_idx} T-Action: {clamped_teacher_actions[target_env_idx].cpu().numpy().round(3)}")
                print(f"  Env {target_env_idx} S-Action: {student_actions[target_env_idx].detach().cpu().numpy().round(3)}")
                
        # --- POV Video Recording ---
        if args_cli.video:
            is_recording_phase = (iteration % args_cli.video_interval) < args_cli.video_length
            
            if is_recording_phase:
                # camera_low.data.output["rgb"] is bounded [0, 255] uint8, Shape: (num_envs, H, W, 3)
                # We record the environment closest to the world center
                frame = rgb_raw[target_env_idx].detach().cpu()
                on_robot_video_frames.append(frame)
            
            # Save when phase ends or at the very last iteration
            if (not is_recording_phase or iteration == args_cli.max_iterations - 1) and len(on_robot_video_frames) >= 2:
                # We finished a recording segment, save to disk
                pov_dir = os.path.join("logs", "bc_runs", args_cli.run_name, "videos_pov")
                os.makedirs(pov_dir, exist_ok=True)
                pov_path = os.path.join(pov_dir, f"pov_iter_{iteration}.mp4")
                
                # Stack to (T, H, W, 3)
                video_tensor = torch.stack(on_robot_video_frames)
                try:
                    torchvision.io.write_video(pov_path, video_tensor, fps=50) # Assuming 50Hz control / step dt
                    print(f"[INFO] Saved POV Video: {pov_path} (Frames: {len(on_robot_video_frames)})")
                except Exception as e:
                    print(f"[WARNING] Failed to save POV video: {e}")
                    
                on_robot_video_frames.clear()

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
