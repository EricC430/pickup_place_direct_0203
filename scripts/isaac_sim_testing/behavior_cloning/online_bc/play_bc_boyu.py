import torch
import torch.nn as nn
import copy
import math
import argparse
import os
import cv2
import numpy as np

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Setup App Launcher before other imports! Critical for Omniverse initialization.
parser = argparse.ArgumentParser(description="Evaluate Online Behavioral Cloning Policy")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run.")
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
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0313_env_cfg import PickupPlaceVisionAsym0313EnvCfg
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0313_env import PickupPlaceVisionAsym0313Env

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
        pass

    def load_states(self, state_dict):
        self.load_state_dict(state_dict, strict=False)

def debug_visualize_coords(images_tensor, gt_coords, pred_coords, step, batch_idx=0, save_dir="eval_vision_debug"):
    """
    Project ground truth (Green) and predicted (Red) coordinates onto the image for debugging.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    img = images_tensor[batch_idx].detach().cpu().numpy().transpose(1, 2, 0) # (H, W, 3)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    h, w = img.shape[:2]
    
    # Map coordinates from [-1, 1] to [0, W/H]
    # gt
    gt_px = int((gt_coords[batch_idx, 0].item() + 1.0) / 2.0 * w)
    gt_py = int((gt_coords[batch_idx, 1].item() + 1.0) / 2.0 * h)
    # pred
    pr_px = int((pred_coords[batch_idx, 0].item() + 1.0) / 2.0 * w)
    pr_py = int((pred_coords[batch_idx, 1].item() + 1.0) / 2.0 * h)
    
    # Draw Ground Truth (Green Circle)
    cv2.circle(img, (gt_px, gt_py), radius=3, color=(0, 255, 0), thickness=-1)
    
    # Draw Prediction (Red Cross)
    cv2.drawMarker(img, (pr_px, pr_py), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
    
    # Text info
    cv2.putText(img, f"GT: ({gt_px}, {gt_py})", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.putText(img, f"PR: ({pr_px}, {pr_py})", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    save_path = os.path.join(save_dir, f"eval_vision_step_{step}.jpg")
    cv2.imwrite(save_path, img)

class StudentActor(nn.Module):
    def __init__(self, vision_encoder_low, pointnet, input_dim=1130, action_dim=6):
        super().__init__()
        # Embed the vision encoders into the model so they get saved!
        self.vision_encoder_low = copy.deepcopy(vision_encoder_low)
        self.pointnet = copy.deepcopy(pointnet)
        
        # Internal clean memory buffers (initialized in forward pass)
        self.vision_low_history_buf = None
        self.pointnet_history_buf = None
        
        # 1. Proprio Normalization
        self.proprio_norm = EmpiricalNormalizer(42)
        
        # 2. Vision Normalization: Separate LayerNorms
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
        
    def forward(self, policy_obs, rgb_raw, ptcloud_world, reset_buf=None):
        # 1. Preprocess RGB
        rgb_norm = rgb_raw[..., :3] / 255.0
        rgb_norm = torch.clamp(rgb_norm, 0.0, 1.0)
        rgb_chw = rgb_norm.permute(0, 3, 1, 2).float()
        
        # 2. Forward CNN & PointNet
        # [THE BLUEPRINT] Vision Encoder now returns (features, predicted_coords)
        vision_low_current, predicted_coords = self.vision_encoder_low(rgb_chw)
        pointnet_current = self.pointnet(ptcloud_world)
        
        # 3. Extract pure Proprio and Static Context
        proprio_obs = policy_obs[:, :42]
        # In eval mode, we don't update normalization stats
        proprio_normed = self.proprio_norm(proprio_obs)
        
        vision_high = policy_obs[:, 1066:]
        
        # 4. Manage CLEAN internal history buffers (Replacing Poisoned Env History)
        B = policy_obs.shape[0]
        device = policy_obs.device
        
        if self.vision_low_history_buf is None or self.vision_low_history_buf.shape[0] != B:
            self.vision_low_history_buf = torch.zeros((B, 512), device=device)
            self.pointnet_history_buf = torch.zeros((B, 512), device=device)
            
        # Reset memory for finished environments
        if reset_buf is not None:
            self.vision_low_history_buf[reset_buf] = 0.0
            self.pointnet_history_buf[reset_buf] = 0.0
            
        # 🚨 [FIX] Detach history buffers to prevent Truncated BPTT backward errors!
        self.vision_low_history_buf = self.vision_low_history_buf.detach()
        self.pointnet_history_buf = self.pointnet_history_buf.detach()
        
        # FIFO Shift Left
        self.vision_low_history_buf = torch.roll(self.vision_low_history_buf, shifts=-128, dims=1)
        self.vision_low_history_buf[:, -128:] = vision_low_current
        
        self.pointnet_history_buf = torch.roll(self.pointnet_history_buf, shifts=-128, dims=1)
        self.pointnet_history_buf[:, -128:] = pointnet_current
        
        vision_low_history = self.vision_low_history_buf.clone()
        pointnet_history = self.pointnet_history_buf.clone()
        
        # 5. Apply LayerNorms
        vision_low_normed = self.vision_low_ln(vision_low_history)
        pointnet_normed = self.pointnet_ln(pointnet_history)
        vision_high_normed = self.vision_high_ln(vision_high)
        
        # 6. MLP Pass
        x = torch.cat([proprio_normed, vision_low_normed, pointnet_normed, vision_high_normed], dim=-1)
        actions = self.mlp(x)
        return actions, predicted_coords

def main():
    print("[INFO] Starting BC Evaluation Script...")
    
    env_cfg = PickupPlaceVisionAsym0313EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    render_mode = "rgb_array" if args_cli.video else None
    env = PickupPlaceVisionAsym0313Env(cfg=env_cfg, render_mode=render_mode)
    
    if args_cli.video:
        import gymnasium as gym
        model_dir = os.path.dirname(args_cli.model_path)
        video_dir = os.path.join(model_dir if model_dir else ".", "eval_videos")
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
    obs, _ = env.reset()
    student_dim = obs["policy"].shape[-1]
    
    print(f"[INFO] Loading trained weights from {args_cli.model_path}")
    student_actor = StudentActor(
        vision_encoder_low=env.unwrapped.vision_encoder_low,
        pointnet=env.unwrapped.pointnet,
        input_dim=student_dim
    ).to(device)
    
    if os.path.exists(args_cli.model_path):
        try:
            ckpt = torch.load(args_cli.model_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            student_actor.load_state_dict(state_dict)
            print("[INFO] Model weights loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load weights: {e}")
            print("[Info] Attempting partial load...")
            ckpt = torch.load(args_cli.model_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            student_actor.load_state_dict(state_dict, strict=False)
    else:
        print(f"[ERROR] Could not find {args_cli.model_path}. Exiting.")
        return
        
    student_actor.eval()
    
    # --- Viewer Camera Tracking (from Blueprint Fix) ---
    env_origins = env.unwrapped.scene.env_origins
    distances = torch.norm(env_origins, dim=-1)
    target_env_idx = torch.argmin(distances).item()
    
    robot_root_w = env.unwrapped.scene["robot"].data.root_pos_w[target_env_idx]
    lookat_pos = robot_root_w.cpu().numpy().tolist()
    # Using the user-requested widened view [1.5, 1.0, 1.2]
    eye_pos = (robot_root_w + torch.tensor([1.5, 1.0, 1.2], device=device)).cpu().numpy().tolist()
    
    if hasattr(env.unwrapped, "sim") and hasattr(env.unwrapped.sim, "set_camera_view"):
        env.unwrapped.sim.set_camera_view(eye=eye_pos, target=lookat_pos)
        print(f"[INFO] POV Evaluation centered on robot at {lookat_pos}")

    print("[INFO] Starting visualization loop. Press Ctrl+C to stop.")
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                policy_obs = obs["policy"]
                rgb_raw = env.unwrapped.camera_low.data.output["rgb"]
                ptcloud_world = env.unwrapped.current_ptcloud
                reset_buf = env.unwrapped.reset_buf
                
                student_actions, predicted_coords = student_actor(policy_obs, rgb_raw, ptcloud_world, reset_buf)
                clamped_actions = torch.clamp(student_actions, -1.0, 1.0)
                
                # Projection Math for Evaluation Visual Debugging
                ue = env.unwrapped
                target_pos_w = ue.scene["object"].data.root_com_pose_w[:, :3]
                cam_pos_w = ue.camera_low.data.pos_w 
                cam_quat_w = ue.camera_low.data.quat_w_world
                
                from isaaclab.utils.math import subtract_frame_transforms
                obj_pos_cam, _ = subtract_frame_transforms(cam_pos_w, cam_quat_w, target_pos_w)
                
                z_opt = -obj_pos_cam[:, 0]
                x_opt = -obj_pos_cam[:, 1]
                y_opt = -obj_pos_cam[:, 2]
                
                H, W = rgb_raw.shape[1], rgb_raw.shape[2] 
                hfov_rad = math.radians(79.0) # target_hfov
                fx = W / (2.0 * math.tan(hfov_rad / 2.0))
                fy = fx
                cx, cy = W / 2.0, H / 2.0
                
                x_img = (fx * x_opt / (z_opt + 1e-6)) + cx
                y_img = (fy * y_opt / (z_opt + 1e-6)) + cy
                
                gt_x = (x_img / W) * 2.0 - 1.0
                gt_y = (y_img / H) * 2.0 - 1.0
                gt_s = torch.clamp(z_opt / 2.5, 0.0, 1.0) * 2.0 - 1.0
                gt_coords = torch.stack([gt_x, gt_y, gt_s], dim=-1)
                
                # 🚨 [新增] 夾爪二元化閘門 (大於 0 張開，小於等於 0 夾緊)
                clamped_actions[:, 5] = torch.where(clamped_actions[:, 5] > 0.0, 1.0, -1.0)
                
            obs, reward, terminated, truncated, extras = env.step(clamped_actions)
            
            step_count = getattr(env.unwrapped, 'common_step_counter', 0)
            if step_count % 50 == 0:
                print(f"[{step_count}] Action magnitude: {student_actions[0].norm().item():.3f}")
                print(f"[{step_count}] GT Pixel: ({x_img[0].item():.1f}, {y_img[0].item():.1f}) | Norm GT: ({gt_x[0].item():.2f}, {gt_y[0].item():.2f})")
                
                # Save visual debug image
                rgb_norm_batch = (rgb_raw.float() / 255.0).permute(0, 3, 1, 2)
                debug_visualize_coords(rgb_norm_batch, gt_coords, predicted_coords, step_count)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Loop crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Closing environment...")
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
