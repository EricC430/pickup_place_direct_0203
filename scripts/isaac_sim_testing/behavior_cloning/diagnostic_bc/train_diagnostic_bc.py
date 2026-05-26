import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from pathlib import Path

# Isaac Lab imports
from isaaclab.app import AppLauncher

# Setup App Launcher
parser = argparse.ArgumentParser(description="Diagnostic State-Based Behavioral Cloning")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments.")
parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum number of BC iterations.")
parser.add_argument("--run_name", type=str, default="diagnostic_bc", help="Name of the run.")
parser.add_argument("--save_interval", type=int, default=1000, help="Save interval.")
parser.add_argument("--diagnose_variance", action="store_true", help="Perform expert dataset variance diagnostic instead of training.")
parser.add_argument("--diagnose_steps", type=int, default=500, help="Number of rollout steps to collect for diagnostic.")
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

    def update(self, x):
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


class DiagnosticActor(nn.Module):
    """
    MLP-only actor for diagnostic state-based BC.
    Input Dim: 72 (Proprio 42 + GT 30)
    """
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
        if self.training:
            self.norm.update(x)
        x_norm = self.norm(x)
        return self.mlp(x_norm)


def get_teacher_obs(env):
    ue = env.unwrapped
    TEACHER_DEFAULT_JPOS = torch.tensor([0.0, 0.61086472, 0.7853975, 0.95993027, 0.0], device=ue.device)
    TEACHER_DEFAULT_JVEL = torch.zeros(5, device=ue.device)
    jpos_arm = ue.joint_pos[:, ue._arm_joint_indices] - TEACHER_DEFAULT_JPOS
    jvel_arm = ue.joint_vel[:, ue._arm_joint_indices] - TEACHER_DEFAULT_JVEL
    obj_pos = mdp_obs.object_position_in_robot_root_frame(ue, object_cfg=SceneEntityCfg("object"))
    bbox = mdp_obs.object_bbox_corners_relative(ue, object_cfg=SceneEntityCfg("object"))
    target_pos_w = ue.target_poses + ue.scene.env_origins
    target, _ = subtract_frame_transforms(ue.scene["robot"].data.root_pos_w, ue.scene["robot"].data.root_quat_w, target_pos_w)
    return torch.cat([jpos_arm, jvel_arm, obj_pos, bbox, target, ue.actions], dim=-1)


def get_current_action_equivalent(ue):
    """
    Reverse mapping: physical joint positions -> [-1, 1] normalized action space.
    This uses the action_cfg parameters from PickupPlaceDirect0203EnvCfg.
    """
    # 臂部 offset 為 0，scale 為 2.09
    arm_q = ue.joint_pos[:, ue._arm_joint_indices]
    arm_actions = arm_q / 2.09
    
    # 夾爪 offset 為 0.785，scale 為 0.785
    # 注意: 這邊假設 r_joint 是代表夾爪的關鍵維度 (與 env.step 對齊)
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
    proprio = torch.cat([jpos_full, jvel_full, jerr, last_4_actions], dim=-1) # 42
    
    # GT Object (30) - relative to EE for better generalization
    ee_pos_w = ue.ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ue.ee_frame.data.target_quat_w[..., 0, :]
    obj_pos_w = ue.scene["object"].data.root_com_pose_w[:, :3]
    obj_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, obj_pos_w) # 3
    target_pos_w = ue.target_poses + ue.scene.env_origins 
    target_pos_ee, _ = subtract_frame_transforms(ee_pos_w, ee_quat_w, target_pos_w) # 3
    world_corners_flat = mdp_obs.object_bbox_corners(ue, SceneEntityCfg("object")).view(ue.num_envs * 8, 3)
    ee_pos_rep = ee_pos_w.repeat_interleave(8, dim=0)
    ee_quat_rep = ee_quat_w.repeat_interleave(8, dim=0)
    bbox_ee_flat, _ = subtract_frame_transforms(ee_pos_rep, ee_quat_rep, world_corners_flat)
    bbox_ee = bbox_ee_flat.view(ue.num_envs, 24) # 24
    
    return torch.cat([proprio, obj_pos_ee, bbox_ee, target_pos_ee], dim=-1) # 42 + 3 + 24 + 3 = 72


def run_variance_diagnostic(env, teacher_actor, num_steps=500):
    """
    Expert Dataset Variance Diagnostic:
    Checks if similar 72-dim states lead to similar expert actions.
    Large variance indicates 'Causal Confusion' or missing state information.
    """
    all_states = []
    all_actions = []

    print(f"[Diagnostic] Starting Rollout Data Collection for {num_steps} steps...", flush=True)
    # env.reset()  <-- Remove redundant reset, main() already does it
    for i in range(num_steps):
        # 1. Get current state and expert action
        diag_obs = get_diagnostic_obs(env.unwrapped)
        priv_obs = get_teacher_obs(env)
        
        with torch.no_grad():
            teacher_actions = teacher_actor(priv_obs)
            clamped_actions = torch.clamp(teacher_actions, -1.0, 1.0)
            
        # 2. Store data on CPU to avoid memory pressure
        all_states.append(diag_obs.detach().cpu())
        all_actions.append(clamped_actions.detach().cpu())
        
        # 3. Advance environment
        env.step(clamped_actions)
        
        if (i+1) % 100 == 0:
            print(f"  Collected {i+1}/{num_steps} steps...", flush=True)

    # Flatten tensors [N, D]
    print(f"[Diagnostic] Flattening {len(all_states)} tensors and preparing CPU analysis...", flush=True)
    states_tensor = torch.cat(all_states, dim=0)   # [steps * num_envs, 72]
    actions_tensor = torch.cat(all_actions, dim=0) # [steps * num_envs, 6]
    total_samples = states_tensor.shape[0]

    # 4. Standardize States (Z-score) on CPU
    # We use CPU to avoid deadlock or extreme slowness when GPU is busy with Isaac Sim cameras
    mean = states_tensor.mean(dim=0)
    std = states_tensor.std(dim=0) + 1e-8
    states_norm = (states_tensor - mean) / std
    print(f"[Diagnostic] Standardization complete on CPU.", flush=True)

    # 5. K-Nearest Neighbors Analysis (Batched CPU approach)
    num_anchors = 1000
    batch_size = 100
    K = 50
    print(f"[Diagnostic] Performing Batched KNN Analysis on CPU (Anchors={num_anchors}, K={K})...", flush=True)
    
    all_anchor_stds = []
    
    with torch.no_grad():
        for i in range(0, num_anchors, batch_size):
            curr_batch_size = min(batch_size, num_anchors - i)
            # Pick unique random anchors for this batch
            indices = torch.randperm(total_samples)[:curr_batch_size]
            anchor_batch = states_norm[indices] # [batch, 72]
            
            # Compute distances: [batch_size, total_samples]
            # torch.cdist is multithreaded and efficient on CPU
            dists = torch.cdist(anchor_batch, states_norm) # [batch, 64000]
            
            # Get Top-K closest neighbors
            _, topk_indices = torch.topk(dists, k=K, dim=1, largest=False) # [batch, K]
            
            # Gather actions for these neighbors
            # neighbor_actions: [batch, K, 6]
            neighbor_actions = actions_tensor[topk_indices]
            
            # Compute standard deviation across K neighbors for each anchor
            batch_stds = neighbor_actions.std(dim=1) # [batch, 6]
            all_anchor_stds.append(batch_stds)
            
            print(f"  Processed {i + curr_batch_size}/{num_anchors} anchors...", flush=True)

        # Average across all 1000 anchors
        print(f"[Diagnostic] Analysis complete. Concatenating results...", flush=True)
        avg_stds = torch.cat(all_anchor_stds, dim=0).mean(dim=0) # [6]
    
    print("\n" + "="*50, flush=True)
    print("      DATASET VARIANCE DIAGNOSTIC REPORT", flush=True)
    print("="*50, flush=True)
    print(f"Total Samples: {total_samples}", flush=True)
    print(f"Avg Action Std Dev (N={num_anchors}, K={K}):", flush=True)
    dims = ["J1", "J2", "J3", "J4", "J5", "Grip"]
    for name, val in zip(dims, avg_stds):
        print(f"  {name:4}: {val.item():.4f}", flush=True)
    
    arm_avg = avg_stds[:5].mean().item()
    grip_avg = avg_stds[5].item()
    
    print("-" * 50, flush=True)
    print(f"Overall Arm Avg Std:      {arm_avg:.4f}", flush=True)
    print(f"Overall Gripper Avg Std:  {grip_avg:.4f}", flush=True)
    
    print("\nINTERPRETATION:", flush=True)
    print(" - Action Range is [-1.0, 1.0] (total width 2.0).", flush=True)
    print(" - Std > 0.2 (10% range) suggests severe Causal Confusion or missing features.", flush=True)
    print(" - Std < 0.05 (2.5% range) suggests the 72-dim state is very consistent.", flush=True)
    print("="*50 + "\n", flush=True)
    
    print("[INFO] Diagnostic finished. Shutting down environment...", flush=True)
    env.close()


def main():
    env_cfg = PickupPlaceVisionAsym0310EnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    print(f"[INFO] Initializing Environment ({args_cli.num_envs} envs)... this may take a moment.", flush=True)
    env = PickupPlaceVisionAsym0310Env(cfg=env_cfg)
    device = env.unwrapped.device
    
    teacher_path = "/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/proprioception_only/2026-02-12_00-12-19/exported/policy.pt"
    if os.path.exists(teacher_path):
        print(f"[INFO] Loading Teacher model: {teacher_path}", flush=True)
        teacher_actor = torch.jit.load(teacher_path).to(device)
        print(f"[INFO] Successfully loaded Teacher.", flush=True)
    else:
        raise FileNotFoundError(f"Teacher not found at {teacher_path}")

    teacher_actor.eval()
    for param in teacher_actor.parameters():
        param.requires_grad = False
        
    print("[INFO] Resetting Environment for the first time...", flush=True)
    obs, _ = env.reset()
    print("[INFO] Environment Ready.", flush=True)

    student_actor = DiagnosticActor(input_dim=72).to(device)
    optimizer = optim.Adam(student_actor.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.05)
    
    if args_cli.diagnose_variance:
        run_variance_diagnostic(env, teacher_actor, num_steps=args_cli.diagnose_steps)
        simulation_app.close()
        return

    print("Starting Diagnostic Delta-Action Training Loop...")
    for iteration in range(args_cli.max_iterations):
        privileged_obs = get_teacher_obs(env)
        with torch.no_grad():
            teacher_actions = teacher_actor(privileged_obs)
        clamped_teacher_actions = torch.clamp(teacher_actions, -1.0, 1.0)
        
        # 獲取當前關節在 Action Space 中的等效值
        current_action_equivalent = get_current_action_equivalent(env.unwrapped)
        
        # 計算 Teacher 目標相對於當前位置的「變動量」
        # target_delta = A_target - A_curr
        target_delta = clamped_teacher_actions - current_action_equivalent
        
        # State-based Student Observation
        diag_obs = get_diagnostic_obs(env.unwrapped)
        
        # Student 現在預測的是 Delta
        predicted_delta = student_actor(diag_obs)

        # 損失函數直接優化 Delta 的擬合
        loss = loss_fn(predicted_delta, target_delta)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 執行環境模擬時，可以使用 Teacher 的絕對動作 (保持軌跡穩定)
        # 或者使用 Student 的 Delta 動作來測試即時閉環效果
        # 此處維持使用 Teacher Action 以確保數據收集的品質
        obs, reward, terminated, truncated, extras = env.step(clamped_teacher_actions)

        if (iteration + 1) % 100 == 0 or iteration < 10:
            l1_error = torch.abs(predicted_delta - target_delta).mean(dim=0)
            print(f"Iter [{iteration+1}/{args_cli.max_iterations}] - Loss: {loss.item():.6f} | L1 ΔArm: {l1_error[:5].mean():.4f} | L1 ΔGrip: {l1_error[5]:.4f}", flush=True)

        if (iteration + 1) % args_cli.save_interval == 0:
            save_path = f"logs/bc_runs/{args_cli.run_name}/model_iter_{iteration+1}.pt"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(student_actor.state_dict(), save_path)
            
    simulation_app.close()

if __name__ == "__main__":
    main()
