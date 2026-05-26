import argparse
import os
import torch
import torch.nn as nn
import copy

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Setup App Launcher
parser = argparse.ArgumentParser(description="Evaluate Diagnostic State-Based Behavioral Cloning Policy")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to run.")
parser.add_argument("--model_path", type=str, default="diagnostic_actor.pt", help="Path to the trained diagnostic model (.pt file)")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval (in steps) between recording videos.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

# Environment imports
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.mdp import observations as mdp_obs
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0310_env_cfg import PickupPlaceVisionAsym0310EnvCfg
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0310_env import PickupPlaceVisionAsym0310Env

class EmpiricalNormalizer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(shape))
        self.register_buffer("running_var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1.0))
        self.epsilon = 1e-8

    def forward(self, x):
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

class DiagnosticActor(nn.Module):
    def __init__(self, input_dim=72, action_dim=6):
        super().__init__()
        self.norm = EmpiricalNormalizer(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim)
        )
        
    def forward(self, x):
        x_norm = self.norm(x)
        return self.mlp(x_norm)

def get_current_action_equivalent(ue):
    """
    Reverse mapping: physical joint positions -> [-1, 1] normalized action space.
    """
    arm_q = ue.joint_pos[:, ue._arm_joint_indices]
    arm_actions = arm_q / 2.09
    gripper_q = ue.joint_pos[:, ue._gripper_joint_idx]
    gripper_actions = (gripper_q - 0.785) / 0.785
    return torch.cat([arm_actions, gripper_actions], dim=-1)


def get_diagnostic_obs(ue):
    # Proprio (42)
    jpos_full = ue.joint_pos[:, list(ue._arm_joint_indices) + list(ue._gripper_joint_idx)] - ue.robot.data.default_joint_pos[:, list(ue._arm_joint_indices) + list(ue._gripper_joint_idx)]
    jvel_full = ue.joint_vel[:, list(ue._arm_joint_indices) + list(ue._gripper_joint_idx)] - ue.robot.data.default_joint_vel[:, list(ue._arm_joint_indices) + list(ue._gripper_joint_idx)]
    prev_actions = ue.action_history_buf[:, -2, :] if ue.action_history_buf.shape[1] >= 2 else ue.actions
    jerr = prev_actions - jpos_full
    last_4_actions = ue.action_history_buf.view(ue.num_envs, -1)
    proprio = torch.cat([jpos_full, jvel_full, jerr, last_4_actions], dim=-1)
    
    # GT Object (30)
    ee_pos_w = ue.ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ue.ee_frame.data.target_quat_w[..., 0, :]
    obj_pos_w = ue.scene["object"].data.root_com_pose_w[:, :3]
    obj_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, obj_pos_w)
    target_pos_w = ue.target_poses + ue.scene.env_origins 
    target_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, target_pos_w)
    world_corners_flat = mdp_obs.object_bbox_corners(ue, SceneEntityCfg("object")).view(ue.num_envs * 8, 3)
    ee_pos_rep = ee_pos_w.repeat_interleave(8, dim=0)
    ee_quat_rep = ee_quat_w.repeat_interleave(8, dim=0)
    bbox_ee_flat, _ = subtract_frame_transforms(ee_pos_rep, ee_quat_rep, world_corners_flat)
    bbox_ee = bbox_ee_flat.view(ue.num_envs, 24)
    
    return torch.cat([proprio, obj_pos_ee, bbox_ee, target_pos_ee], dim=-1)

def main():
    print("[INFO] Starting Diagnostic BC Evaluation Script...")
    env_cfg = PickupPlaceVisionAsym0310EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    render_mode = "rgb_array" if args_cli.video else None
    env = PickupPlaceVisionAsym0310Env(cfg=env_cfg, render_mode=render_mode)
    
    # Optional Video Recorder
    if args_cli.video:
        import gymnasium as gym
        model_dir = os.path.dirname(args_cli.model_path) or "."
        video_dir = os.path.join(model_dir, "eval_videos_diag")
        os.makedirs(video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True
        )
        
    device = env.unwrapped.device
    obs, _ = env.reset()
    
    print(f"[INFO] Loading diagnostic weights from {args_cli.model_path}")
    student_actor = DiagnosticActor(input_dim=72).to(device)
    
    if os.path.exists(args_cli.model_path):
        student_actor.load_state_dict(torch.load(args_cli.model_path, map_location=device))
        print("[INFO] Diagnostic model weights loaded successfully!")
    else:
        print(f"[ERROR] Could not find {args_cli.model_path}. Exiting.")
        return
        
    student_actor.eval()
    
    print("[INFO] Starting visualization loop. Press Ctrl+C to stop.")
    try:
        while simulation_app.is_running():
            with torch.no_grad():
                # 獲取環境當前狀態與觀察
                diag_obs = get_diagnostic_obs(env.unwrapped)
                
                # 計算當前位置在 Action Space 中的等效值
                current_action_equivalent = get_current_action_equivalent(env.unwrapped)
                
                # Student 預測 Delta
                student_delta = student_actor(diag_obs)
                
                # 最終動作 = 當前位置 + 變動量，並限制在 [-1, 1]
                final_actions = torch.clamp(current_action_equivalent + student_delta, -1.0, 1.0)
                
            obs, reward, terminated, truncated, extras = env.step(final_actions)
            
            if env.unwrapped.common_step_counter % 50 == 0:
                print(f"[{env.unwrapped.common_step_counter}] Action magnitude: {final_actions[0].norm().item():.3f} | Delta magnitude: {student_delta[0].norm().item():.3f}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
