# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint and record the successful trajectory for Sim2Real.
   Adapted for the 0510 Task-Space Delta IK environment (7D action, 49D obs).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import torch
import numpy as np

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Record Sim2Real trajectory (0510 Delta IK).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Pickup-Place-Direct-0510-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--object_id", type=int, default=None, help="The specific object ID to spawn for testing.")
parser.add_argument("--real_obj_pos", action="store_true", default=False, help="Use real-time object position at every frame instead of locking it.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Force PyTorch CUDA context initialization
try:
    if torch.cuda.is_available():
        _ = torch.zeros(1, device="cuda")
except Exception as e:
    pass

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time

from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

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
    """Play with RSL-RL agent and record trajectory (0510 Delta IK version)."""
    # Ensure 1 environment for isolated recording
    env_cfg.scene.num_envs = 1
    
    # 0510 decimation=10, dt=0.01 -> step_dt=0.1s, episode=5s -> 50 steps per episode
    max_steps = 50
    EPISODE_NUM = 3
    
    # Override object ID if configured via CLI
    if args_cli.object_id is not None:
        obj_id = args_cli.object_id
        print(f"\n[INFO] 🎯 Configuring environment to spawn ONLY object ID: {obj_id}")
        
        # 1. Modify the config spawner assets
        from isaaclab.sim.spawners.from_files import UsdFileCfg
        if hasattr(env_cfg, "object_cfg"):
            env_cfg.object_cfg.spawn.assets_cfg = [
                UsdFileCfg(
                    usd_path=f"/workspace/test_isaaclab/ObjectFolder_selected/{obj_id}/{obj_id}.usd",
                    scale=(0.6, 0.6, 0.6),
                )
            ]
            
        # 2. Modify SELECTED_OBJECT_IDS in-place across all loaded config modules to update internal lists
        import sys
        found_mod = False
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.endswith("pickup_place_direct_0510_env_cfg") and mod is not None:
                if hasattr(mod, "SELECTED_OBJECT_IDS"):
                    mod.SELECTED_OBJECT_IDS.clear()
                    mod.SELECTED_OBJECT_IDS.append(obj_id)
                    found_mod = True
        if not found_mod:
            try:
                import pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_direct_0510_env_cfg as env_cfg_mod
                env_cfg_mod.SELECTED_OBJECT_IDS.clear()
                env_cfg_mod.SELECTED_OBJECT_IDS.append(obj_id)
            except Exception as e:
                print(f"[WARNING] Failed to update SELECTED_OBJECT_IDS: {e}")
    
    # 0510 uses 49D observation space (use_46_dim_obs = False)
    if hasattr(env_cfg, "use_46_dim_obs"):
        env_cfg.use_46_dim_obs = False
    env_cfg.observation_space = 49

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # Adjust camera position for recording
    env_cfg.viewer.eye = [1.0, -1.0, 1.0]
    env_cfg.viewer.lookat = [0.15, 0.0, 0.1]
    
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # 使用 Isaac Lab 原生方法強制設定相機視角
    if not args_cli.headless:
        env.unwrapped.sim.set_camera_view(eye=[1.0, -1.0, 1.0], target=[0.15, 0.0, 0.1])
        
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        env.unwrapped.sim.render_mode = env.unwrapped.sim.RenderMode.PARTIAL_RENDERING
        
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(resume_path), "videos", "play_0510"),
            "step_trigger": lambda step: step == 0,
            "video_length": EPISODE_NUM * max_steps,  # Adapt video length to configured episodes
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
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
    except Exception as e:
        runner.load(resume_path)

    # ==== Deterministic inference ====
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs, _ = env.reset()
    
    # 預先查詢夾爪關節索引
    _gripper_joint_idx, _ = env.unwrapped.scene["robot"].find_joints("r_joint")

    # Tensors for recording
    recorded_obj_pos = []
    recorded_policy_actions = []
    recorded_robot_joint_targets = []
    
    # 獲取受控關節索引 (手臂 + 夾爪)
    arm_joint_indices = env.unwrapped._arm_joint_indices
    gripper_joint_idx = env.unwrapped._gripper_joint_idx
    controlled_joint_ids = list(arm_joint_indices) + list(gripper_joint_idx)

    print("\n[INFO] Starting Simulation & Recording (0510 Delta IK)...")
    print(f"[INFO] 🤖 受控關節: {len(controlled_joint_ids)} 個 (手臂5 + 夾爪1)")
    print(f"[INFO] 🎮 動作空間: 7D (dx, dy, dz, drx, dry, drz, gripper)")
    print(f"[INFO] 👁️ 觀測空間: 49D")

    # simulate environment
    step_count = 0

    # 宣告記憶變數：儲存初始數據與夾取狀態
    init_pos = None
    init_bbox = None
    
    # Delta Kinematics 參數
    GRASP_ANGLE_THRESHOLD = 0.5
    is_grasped = False
    grasp_moment_ee_pos = None
    
    # 觀測值記錄器
    recorded_obs_list = []
    OBS_RECORD_LIMIT = 50
    
    # 去彈跳與高度防呆參數
    grasp_counter = 0
    TARGET_GRASP_STEPS = 5  # 0510 步數較少，降低門檻
    DISTANCE_THRESHOLD = 0.10
    dynamic_z_threshold = None

    WARMUP_STEPS = 5  # 0510 decimation=10，每步 0.1s，5 步 = 0.5s 暖機
    warmup_joint_pos = None

    i = 0
    while simulation_app.is_running() and i < EPISODE_NUM:
        print(f"\n==================== [Episode {i+1} / {EPISODE_NUM}] ====================")
        with torch.inference_mode():
            obs, _ = env.reset()
        
        # 重置每回合狀態以防不同回合之間的記憶殘留與漏失 (Fix object penetration and miss issues)
        init_pos = None
        init_bbox = None
        is_grasped = False
        grasp_moment_ee_pos = None
        grasp_counter = 0
        dynamic_z_threshold = None
        warmup_joint_pos = None
        
        step_count = 0
        while simulation_app.is_running() and step_count < max_steps:
            with torch.inference_mode():
                # 處理 TensorDict / Dict / Tensor 的相容性
                obs_tensor = obs["policy"] if hasattr(obs, "keys") and "policy" in obs.keys() else obs

                # ==========================================
                # 暖機攔截
                # ==========================================
                if step_count < WARMUP_STEPS:
                    if warmup_joint_pos is None:
                        warmup_joint_pos = env.unwrapped.scene["robot"].data.joint_pos.clone()
                    
                    env.unwrapped.scene["robot"].set_joint_position_target(warmup_joint_pos)
                    
                    zero_actions = torch.zeros_like(policy(obs))
                    obs, _, dones, extras = env.step(zero_actions)
                    
                    recorded_policy_actions.append({
                        'step': step_count,
                        'actions': zero_actions.clone().cpu().numpy().flatten(),
                    })
                    
                    all_joint_targets = env.unwrapped.scene["robot"].data.joint_pos_target.clone()
                    all_joint_pos = env.unwrapped.scene["robot"].data.joint_pos.clone()
                    
                    robot_joint_targets = all_joint_targets[:, controlled_joint_ids].cpu().numpy().flatten()
                    robot_joint_pos = all_joint_pos[:, controlled_joint_ids].cpu().numpy().flatten()
                    
                    recorded_robot_joint_targets.append({
                        'step': step_count,
                        'joint_pos_target': robot_joint_targets,
                        'joint_pos': robot_joint_pos,
                    })
                    
                    if step_count % 2 == 0:
                        current_z = env.unwrapped.scene["object"].data.root_pos_w[0, 2].item()
                        print(f"[WARMUP] Step {step_count:2d}: 讓子彈飛... 物體目前高度 Z = {current_z:.4f}")
                    
                    step_count += 1
                    continue

                # ==========================================
                # 暖機結束
                # ==========================================
                if step_count == WARMUP_STEPS:
                    print("\n[START] 🌍 暖機結束！物體已落地，模型正式啟動！\n")

                # ==== 觀測值數據記錄 ====
                if step_count >= WARMUP_STEPS and len(recorded_obs_list) < OBS_RECORD_LIMIT:
                    recorded_obs_list.append(obs_tensor.clone().cpu())
                    if len(recorded_obs_list) == OBS_RECORD_LIMIT:
                        obs_save_base = os.path.join(os.path.dirname(resume_path), "recorded_obs_50steps")
                        full_obs_tensor = torch.cat(recorded_obs_list, dim=0)  # Shape: (N, 49)
                        
                        torch.save(full_obs_tensor, f"{obs_save_base}.pt")
                        np.savetxt(f"{obs_save_base}.csv", full_obs_tensor.numpy(), delimiter=",")
                        
                        print(f"\n[INFO] 📝 已成功記錄觀測值！")
                        print(f"      -> {obs_save_base}.pt")
                        print(f"      -> {obs_save_base}.csv\n")

                # 1. 首幀資料擷取
                # 0510 (49D): jpos[0:6], jvel[6:12], obj_pos[12:15], bbox[15:39], target[39:42], actions[42:49]
                if init_pos is None:
                    init_pos = obs_tensor[:, 12:15].clone()
                    init_bbox = obs_tensor[:, 15:39].clone()
                    
                    obj_com_w_all = env.unwrapped.scene["object"].data.root_com_pos_w
                    obj_pivot_w_all = env.unwrapped.scene["object"].data.root_pos_w
                    robot_pos_w_all = env.unwrapped.scene["robot"].data.root_pos_w
                    robot_quat_w_all = env.unwrapped.scene["robot"].data.root_quat_w
                    
                    real_com_robot, _ = subtract_frame_transforms(robot_pos_w_all, robot_quat_w_all, obj_com_w_all)
                    real_pivot_robot, _ = subtract_frame_transforms(robot_pos_w_all, robot_quat_w_all, obj_pivot_w_all)
                    
                    dynamic_z_threshold = init_pos[0, 2].item() + 0.10
                    
                    if args_cli.real_obj_pos:
                        print(f"\n[EVAL MODE] 🟢 啟用實時觀測模式 (Real-time Object Position Tracking)！")
                        print(f"📍 初始真實位置 (AI 觀測): {init_pos[0].cpu().numpy()}")
                    else:
                        print(f"\n[BLIND TEST] 🔒 初始數據已鎖死 (Blind Test Mode)！")
                        print(f"📍 鎖定位置 (AI 觀測): {init_pos[0].cpu().numpy()}")
                    print(f"🛡️ 物理真實 (質心 COM):  {real_com_robot[0].cpu().numpy()}")
                    print(f"🛡️ 物理真實 (原點 Pivot): {real_pivot_robot[0].cpu().numpy()}")
                    print(f"📏 動態防呆觸發高度線: < {dynamic_z_threshold:.4f}m\n")
                
                # 2. 擷取夾爪與 EE 當下真實狀態
                gripper_value = env.unwrapped.scene["robot"].data.joint_pos[0, _gripper_joint_idx[0]].item()
                
                ee_pos_w = env.unwrapped.scene["ee_frame"].data.target_pos_w[:, 0, :]
                robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w
                robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w
                
                current_ee_pos, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, ee_pos_w)
                ee_z = current_ee_pos[0, 2].item()
                
                # 3. 去彈跳、高度、距離判定
                obj_pos_w = env.unwrapped.scene["object"].data.root_pos_w[:, :3]
                obj_pos_robot, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, obj_pos_w)
                ee_to_obj_dist = torch.norm(current_ee_pos - obj_pos_robot, dim=1).item()
                
                if gripper_value <= GRASP_ANGLE_THRESHOLD and ee_z < dynamic_z_threshold and ee_to_obj_dist < DISTANCE_THRESHOLD:
                    grasp_counter += 1
                else:
                    grasp_counter = 0
                    
                if grasp_counter >= TARGET_GRASP_STEPS and not is_grasped:
                    is_grasped = True
                    grasp_moment_ee_pos = current_ee_pos.clone()
                    print(f"[BLIND TEST] ✅ 穩定夾取判定成功！啟動跟隨模式 (Gripper={gripper_value:.2f}, EE_Z={ee_z:.4f})")

                # 4. 觀測值強制覆蓋 (Overwrite)
                # 0510 (49D): obj_pos at [12:15], bbox at [15:39]
                if not args_cli.real_obj_pos:
                    if not is_grasped:
                        noise_scale = 0.000 
                        pos_noise = torch.randn_like(init_pos) * noise_scale
                        bbox_noise = torch.randn_like(init_bbox) * noise_scale
                        
                        obs_tensor[:, 12:15] = init_pos + pos_noise
                        obs_tensor[:, 15:39] = init_bbox + bbox_noise
                    else:
                        delta_pos = current_ee_pos - grasp_moment_ee_pos
                        
                        current_obj_pos = init_pos + delta_pos
                        current_obj_bbox = init_bbox.view(-1, 8, 3) + delta_pos.unsqueeze(1)
                        current_obj_bbox = current_obj_bbox.view(-1, 24)
                        
                        obs_tensor[:, 12:15] = current_obj_pos
                        obs_tensor[:, 15:39] = current_obj_bbox

                # agent stepping (deterministic)
                actions = policy(obs)
                
                # 記錄 Policy 輸出的 Actions (存檔用)
                recorded_policy_actions.append({
                    'step': step_count,
                    'actions': actions.clone().cpu().numpy().flatten(),
                })
                
                # ==== Object position recording ====
                robot_pos_w = env.unwrapped.scene["robot"].data.root_pos_w
                robot_quat_w = env.unwrapped.scene["robot"].data.root_quat_w
                obj_pos_w = env.unwrapped.scene["object"].data.root_pos_w
                
                obj_pos_robot, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, obj_pos_w)
                recorded_obj_pos.append(obj_pos_robot[0].cpu().numpy())

                # env stepping
                obs, _, dones, extras = env.step(actions)
                
                # 記錄機器人的實際 Joint Position Targets
                all_joint_targets = env.unwrapped.scene["robot"].data.joint_pos_target.clone()
                all_joint_pos = env.unwrapped.scene["robot"].data.joint_pos.clone()
                
                robot_joint_targets = all_joint_targets[:, controlled_joint_ids].cpu().numpy().flatten()
                robot_joint_pos = all_joint_pos[:, controlled_joint_ids].cpu().numpy().flatten()
                
                # 即時印出模型輸出 (7D Delta IK) 與最終關節目標值對比
                if step_count >= WARMUP_STEPS:
                    raw_action_np = actions[0].clone().cpu().numpy()
                    print(f"[Step {step_count:3d}] 模型原始輸出 (7D): {np.round(raw_action_np, 4)}")
                    print(f"           實際送給關節 (6D): {np.round(robot_joint_targets, 4)}")
                
                recorded_robot_joint_targets.append({
                    'step': step_count,
                    'joint_pos_target': robot_joint_targets,
                    'joint_pos': robot_joint_pos,
                })
                
                # ==== 環境重置處理 ====
                if dones.any():
                    print("\n[BLIND TEST] 🔓 環境重置，任務結束！")
                    
                    # 取得重置原因與最後狀態以進行 Debug
                    time_out = False
                    if "time_outs" in extras:
                        time_out = extras["time_outs"].any().item()
                    
                    print("  [重置分析]")
                    if time_out:
                        print("    - 觸發條件: 時間截止 (Time Out / Truncated) - 已達最大步數限制")
                    else:
                        print("    - 觸發條件: 剛體出界 (Out of Workspace / Terminated) - 物體超出允許的工作空間範圍")
                    
                    # 印出重置前的最後關鍵狀態
                    print("  [重置前最後狀態 (Step {})]".format(step_count))
                    print(f"    - 夾爪位置 (EE Pos): {np.round(current_ee_pos[0].cpu().numpy(), 4)}")
                    print(f"    - 物體位置 (Obj Pos): {np.round(obj_pos_robot[0].cpu().numpy(), 4)}")
                    print(f"    - 夾爪與物體距離: {ee_to_obj_dist:.4f} m")
                    print(f"    - 夾爪開合值: {gripper_value:.4f}")
                    print(f"    - 是否判定為已夾取: {is_grasped}")
                    
                    # 印出環境回傳的最終指標
                    if "episode" in extras:
                        ep_info = extras["episode"]
                        print("  [任務最終指標]")
                        for key in ["reaching_success", "lifting_success", "object_goal_tracking_success"]:
                            if key in ep_info:
                                val = ep_info[key]
                                if hasattr(val, "item"):
                                    val = val.item()
                                print(f"    - {key}: {bool(val)}")
                    break
                else:
                    step_count += 1
        i += 1

    # ==== Save recorded data ====
    if len(recorded_policy_actions) > 0 or len(recorded_robot_joint_targets) > 0:
        import pandas as pd
        
        save_dir = os.path.join(os.path.dirname(resume_path), "policy_inference_logs")
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存 Policy Actions (7D)
        if len(recorded_policy_actions) > 0:
            actions_data = []
            for record in recorded_policy_actions:
                row = {'step': record['step']}
                for i, val in enumerate(record['actions']):
                    row[f'action_{i}'] = val
                actions_data.append(row)
            
            actions_df = pd.DataFrame(actions_data)
            actions_csv_path = os.path.join(save_dir, "policy_actions_0510.csv")
            actions_df.to_csv(actions_csv_path, index=False)
            print(f"\n[INFO] 📊 Policy Actions (7D) 已保存:")
            print(f"      -> {actions_csv_path}")
            print(f"      -> 記錄步數: {len(recorded_policy_actions)}")
        
        # 保存 Robot Joint Targets (6D)
        if len(recorded_robot_joint_targets) > 0:
            targets_data = []
            for record in recorded_robot_joint_targets:
                row = {'step': record['step']}
                for i, val in enumerate(record['joint_pos_target']):
                    row[f'target_{i}'] = val
                for i, val in enumerate(record['joint_pos']):
                    row[f'pos_{i}'] = val
                targets_data.append(row)
            
            targets_df = pd.DataFrame(targets_data)
            targets_csv_path = os.path.join(save_dir, "robot_joint_targets_0510.csv")
            targets_df.to_csv(targets_csv_path, index=False)
            print(f"\n[INFO] 🤖 Robot Joint Targets (6D) 已保存:")
            print(f"      -> {targets_csv_path}")
            print(f"      -> 記錄步數: {len(recorded_robot_joint_targets)}")
            
            print(f"\n[INFO] 前 5 步的數據預覽:")
            print(targets_df.head())

    # close the simulator
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
