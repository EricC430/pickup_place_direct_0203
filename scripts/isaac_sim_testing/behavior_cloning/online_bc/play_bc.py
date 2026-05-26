import argparse
import os
import torch
import torch.nn as nn
import copy

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Setup App Launcher before other imports! Critical for Omniverse initialization.
parser = argparse.ArgumentParser(description="Evaluate Online Behavioral Cloning Policy")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to run.")
parser.add_argument("--model_path", type=str, default="student_actor_bc.pt", help="Path to the trained BC model (.pt file)")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval (in steps) between recording videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

# Environment imports
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0310_env_cfg import PickupPlaceVisionAsym0310EnvCfg
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0310_env import PickupPlaceVisionAsym0310Env

# We must use exactly the same Actor architecture to successfully load the state dict
class EmpiricalNormalizer(nn.Module):
    """
    Normalizes observations using running mean and variance.
    Matches the RSL_RL 'EmpiricalNormalization' behavior.
    """
    def __init__(self, shape):
        super().__init__()
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
        
        # 2. Forward CNN & PointNet
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
        return self.mlp(x)

def main():
    print("[INFO] Starting BC Evaluation Script...")
    
    # 1. Initialize environment (same as training)
    # from isaaclab.envs.wrappers import RecordEnvWrapper
    env_cfg = PickupPlaceVisionAsym0310EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Initialize the environment with rendering if video is requested
    render_mode = "rgb_array" if args_cli.video else None
    env = PickupPlaceVisionAsym0310Env(cfg=env_cfg, render_mode=render_mode)
    
    # 1.5 Wrap with video recorder if requested
    if args_cli.video:
        import gymnasium as gym
        # Save the video in the same directory as the model being evaluated
        model_dir = os.path.dirname(args_cli.model_path)
        if not model_dir:
            model_dir = "."
        video_dir = os.path.join(model_dir, "eval_videos")
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
    
    # 2. Reset the environment
    obs, _ = env.reset()
    student_dim = obs["policy"].shape[-1]
    
    # 3. Initialize the Student Actor and Load Weights
    print(f"[INFO] Loading trained weights from {args_cli.model_path}")
    student_actor = StudentActor(
        vision_encoder_low=env.unwrapped.vision_encoder_low,
        pointnet=env.unwrapped.pointnet,
        input_dim=student_dim
    ).to(device)
    
    if os.path.exists(args_cli.model_path):
        student_actor.load_state_dict(torch.load(args_cli.model_path, map_location=device))
        print("[INFO] Model weights loaded successfully!")
    else:
        print(f"[ERROR] Could not find {args_cli.model_path}. Exiting.")
        return
        
    student_actor.eval() # Set to evaluation mode (e.g. disables dropout if any)
    
    # 4. Evaluation Loop
    print("[INFO] Starting visualization loop. Press Ctrl+C to stop.")
    
    # ---------------- Video Recording (On-Robot POV) ----------------
    on_robot_video_frames = []
    if args_cli.video:
        import torchvision
        
    try:
        while simulation_app.is_running():
            # In evaluation, we do NOT compute gradients
            with torch.no_grad():
                policy_obs = obs["policy"]
                rgb_raw = env.unwrapped.camera_low.data.output["rgb"]
                ptcloud_world = env.unwrapped.current_ptcloud
                
                # Get action from our pure-vision neural network
                student_actions = student_actor(policy_obs, rgb_raw, ptcloud_world)
                
                # Clamp actions to safe bounds [-1.0, 1.0] just like an RL wrapper would do
                clamped_actions = torch.clamp(student_actions, -1.0, 1.0)
                
            # Step the physics environment
            obs, reward, terminated, truncated, extras = env.step(clamped_actions)
            
            # Print actions occasionally for debugging
            if env.unwrapped.common_step_counter % 50 == 0:
                print(f"[{env.unwrapped.common_step_counter}] Action magnitude: {student_actions[0].norm().item():.3f}")
                
            # --- POV Video Recording ---
            if args_cli.video:
                step_count = env.unwrapped.common_step_counter
                is_recording_phase = (step_count % args_cli.video_interval) < args_cli.video_length
                
                if is_recording_phase:
                    # camera_low.data.output["rgb"] is bounded [0, 255] uint8, Shape: (num_envs, H, W, 3)
                    frame = rgb_raw[target_env_idx].detach().cpu()
                    on_robot_video_frames.append(frame)
                
                if (not is_recording_phase or not simulation_app.is_running()) and len(on_robot_video_frames) >= 2:
                    # Save to the same directory as the model
                    model_dir = os.path.dirname(args_cli.model_path)
                    if not model_dir: model_dir = "."
                    pov_dir = os.path.join(model_dir, "eval_videos_pov")
                    os.makedirs(pov_dir, exist_ok=True)
                    pov_path = os.path.join(pov_dir, f"pov_step_{step_count}.mp4")
                    
                    video_tensor = torch.stack(on_robot_video_frames)
                    try:
                        torchvision.io.write_video(pov_path, video_tensor, fps=50) # Assuming 50Hz control / step dt
                        print(f"[INFO] Saved POV Video: {pov_path} (Frames: {len(on_robot_video_frames)})")
                    except Exception as e:
                        print(f"[WARNING] Failed to save POV video: {e}")
                        
                    on_robot_video_frames.clear()

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
