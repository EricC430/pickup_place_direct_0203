from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

from isaaclab.utils.math import subtract_frame_transforms
from .observations import object_bbox_corners


def _get_object_bottom_z(env: "DirectRLEnv", object_cfg: SceneEntityCfg) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]

    if hasattr(env, "object_local_corners"):
        corners_flat = object_bbox_corners(env, object_cfg)
        world_corners = corners_flat.view(env.num_envs, 8, 3)
        world_corners_z = world_corners[..., 2]
        object_bottom_z, _ = torch.min(world_corners_z, dim=1)
    else:
        object_bottom_z = object.data.root_com_pose_w[:, 2] - 0.03

    return object_bottom_z


def object_is_lifted(
    env: "DirectRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    floor_height: float = 0.0,
) -> torch.Tensor:
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    lift_height = object_bottom_z - floor_height
    lifting_reward = lift_height / (minimal_height + 1e-6)
    lifting_reward = torch.clamp(lifting_reward, min=0.0, max=1.0)
    is_settled = env.episode_length_buf > 15
    return lifting_reward * is_settled.float()


def lifting_velocity_reward(env: "DirectRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    obj_vel = env.scene[object_cfg.name].data.root_lin_vel_w[:, 2]
    return torch.clamp(obj_vel, min=0.0, max=1.0)


def gripper_open_close_phases(
    env: "DirectRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    joint_name: str = "r_joint",
    close_threshold: float = 0.05,
    open_target: float = 1.569,
    close_target: float = 0.0,
) -> torch.Tensor:
    """
    Conditional gripper reward with stability bonus (0402 version).
    
    Phase 1 (Approach): dist >= close_threshold → reward openness
    Phase 2 (Grasp):    dist <  close_threshold → reward closedness + stability
    
    Stability Bonus: Rewards the gripper being closed AND low-velocity (stable hold),
                     penalizing high-frequency chattering that prevents real grasps.
    Transition Bonus: One-shot encouragement for initial closing motion.
    """

    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    obj_pos = env.scene[object_cfg.name].data.root_com_pos_w[:, :3]
    dist = torch.norm(obj_pos - ee_pos, dim=-1)

    robot = env.scene["robot"]
    joint_indices, _ = robot.find_joints(joint_name)
    gripper_idx = joint_indices[0]
    gripper_pos = robot.data.joint_pos[:, gripper_idx]
    gripper_vel = robot.data.joint_vel[:, gripper_idx]

    # Openness calculation (1.0 = fully open, 0.0 = fully closed)
    openness = torch.clamp(gripper_pos / open_target, 0.0, 1.0)
    closedness = 1.0 - openness

    # ===== Phase 1: Approach (dist >= threshold) =====
    # Reward keeping gripper open while moving toward the object
    is_approaching = (dist >= close_threshold).float()
    reaching_reward = 1.0 - torch.tanh(dist / 0.10)
    approach_reward = openness * reaching_reward * is_approaching

    # ===== Phase 2: Grasp (dist < threshold) =====
    is_grasping = (dist < close_threshold).float()
    
    # 2a. Base grasp reward: closedness (0=open → 0 pts, 1=closed → 1 pt)
    #     [0403 FIX] Amplified 3x to make closing signal competitive with reaching (~0.5/step)
    #     Without this, grasp reward (~0.15/step) is drowned by reaching and the policy
    #     converges to "hover near object with gripper open" local optimum.
    grasp_reward_base = closedness * 3.0
    
    # 2b. Transition Bonus: +0.2 if gripper is actively closing (vel < -0.1)
    #     [0403 FIX] Increased from 0.1 → 0.2 to reward initial close motion more strongly
    is_closing = (gripper_vel < -0.1).float()
    transition_bonus = 0.2 * is_closing
    
    # 2c. Stability Bonus: Rewards closed + low velocity (stable hold)
    #     Prevents ±7 rad/s chattering. Max value = closedness * 1.0 when vel=0.
    #     NOT overlapping with lifting_bonus: lifting_bonus requires object height > 6cm,
    #     stability_bonus only requires gripper_pos near closed + low vel.
    #     [0403 FIX] Scale 0.3 → 0.5 to further penalize chattering vs stable hold
    vel_stability = 1.0 - torch.clamp(torch.abs(gripper_vel) / 3.0, 0.0, 1.0)
    stability_bonus = closedness * vel_stability * 0.5
    
    grasp_total = (grasp_reward_base + transition_bonus + stability_bonus) * is_grasping

    # [0402 OOM FIX] Reduced print frequency to minimize CPU-GPU sync overhead
    if (env.common_step_counter // 50) % 200 == 0:
        # Compute target gripper pos from previous action to show policy intent
        prev_act = env.previous_actions if hasattr(env, 'previous_actions') else env.actions
        _g_scale = env.cfg.action_cfg["gripper_scale"]   # 0.785
        _g_offset = env.cfg.action_cfg["gripper_offset"]  # 0.785
        _target_grip = (prev_act[:, 5] * env.cfg.action_scale * _g_scale + _g_offset).clamp(0.0, 1.57)
        print(f"Step {env.common_step_counter} env 0: gripper_pos={gripper_pos[0]:.3f}, target={_target_grip[0]:.3f}, vel={gripper_vel[0]:.3f}, dist={dist[0]:.3f}, approach_rew={approach_reward[0]:.3f}, grasp_rew={grasp_total[0]:.3f}, stab={stability_bonus[0]:.3f}")

    return approach_reward + grasp_total

    # ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    # obj_pos = env.scene[object_cfg.name].data.root_com_pos_w[:, :3]
    # dist = torch.norm(obj_pos - ee_pos, dim=-1)

    # robot = env.scene["robot"]
    # joint_indices, _ = robot.find_joints(joint_name)
    # gripper_idx = joint_indices[0]
    # gripper_pos = robot.data.joint_pos[:, gripper_idx]
    # gripper_vel = robot.data.joint_vel[:, gripper_idx]

    # if (env.common_step_counter // 50) % 20 == 0:
    #     approach_phase = (dist > close_threshold)[0].float().item()
    #     grasping_phase = (dist <= close_threshold)[0].float().item()
    #     print(f"Step {env.common_step_counter} env 0: gripper_pos={gripper_pos[0]:.3f}, vel={gripper_vel[0]:.3f}, dist={dist[0]:.3f}, approach_phase={approach_phase}, grasping_phase={grasping_phase}")

    # # Grasp quality: 0-1, where 1 = fully closed (gripper_pos = 0.0)
    # grasp_quality = 1.0 - (gripper_pos / 1.569)  # Clamp to [0, 1]
    # grasp_quality = torch.clamp(grasp_quality, 0.0, 1.0)

    # # Phase 1: Approach phase (dist > close_threshold)
    # # Reward moving gripper fingers toward object
    # approach_reward = (1.0 - torch.tanh(dist / 0.05)) * (dist > close_threshold).float()

    # # Phase 2: Grasping phase (dist <= close_threshold)
    # # Reward stable, slow closing - not rapid/jerky closing
    # is_in_range = (dist <= close_threshold).float()
    
    # # Penalize closing speed (prefer smooth closing, not rapid)
    # # closing_speed = -gripper_vel (negative velocity = closing)
    # closing_speed_penalty = torch.clamp(torch.abs(gripper_vel) / 1.0, 0.0, 1.0)  # 0-1
    
    # # Reward: 
    # # - 40% progress toward full closure
    # # - 40% stable closure speed (low speed)
    # # - 20% maintaining contact
    # grasping_reward = (
    #     0.6 * grasp_quality +                            # Closure progress
    #     0.2 * (1.0 - closing_speed_penalty) +            # Stable speed (penalize high speed)
    #     0.2 * (1.0 - torch.clamp(dist / 0.02, 0.0, 1.0))  # Contact maintenance
    # ) * is_in_range
    
    # return approach_reward + grasping_reward

    # """
    # [DIAGNOSTIC MODE] 
    # Always reward the gripper for being fully open (gripper_pos ~ 1.569).
    # Use this to check if the neural network and action mappings can successfully control the gripper.
    # """
    # robot = env.scene["robot"]
    # joint_indices, _ = robot.find_joints(joint_name)
    # gripper_idx = joint_indices[0]
    # gripper_pos = robot.data.joint_pos[:, gripper_idx]
    # # open_target is ~1.569. Reward is 1.0 when fully open, 0.0 when fully closed.
    # open_quality = torch.clamp(gripper_pos / open_target, 0.0, 1.0)
    # if (env.common_step_counter // 50) % 20 == 0:
    #     print(f"Step {env.common_step_counter} env 0: gripper{gripper_idx}_pos={gripper_pos[0]:.3f}, open_quality={open_quality[0]:.3f}")
    # return open_quality


def object_goal_distance_real(
    env: "DirectRLEnv",
    target_pos: torch.Tensor = None,
    command_name: str = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Compute distance from object to goal position.
    
    In Direct workflow, use target_pos parameter (from env.target_poses).
    In Manager-based workflow, use command_name parameter (deprecated for Direct).
    
    Args:
        env: Environment instance
        target_pos: Target position in world frame (num_envs, 3) - used in Direct workflow
        command_name: Command name (unused in Direct) - kept for backward compatibility
        robot_cfg: Robot configuration
        object_cfg: Object configuration
    
    Returns:
        Distance tensor of shape (num_envs,)
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # For Direct workflow: use target_pos directly
    if target_pos is not None:
        des_pos_w = target_pos
    # For Manager-based workflow (fallback, should not be used in Direct)
    elif command_name is not None:
        try:
            command = env.command_manager.get_command(command_name)
            des_pos_b = command[:, :3]
            des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
        except Exception:
            return torch.full((env.num_envs,), 1e3, device=env.device)
    else:
        # Neither target_pos nor command_name provided
        return torch.full((env.num_envs,), 1e3, device=env.device)
    
    distance = torch.norm(des_pos_w - object.data.root_com_pose_w[:, :3], dim=1)
    return distance


def object_goal_distance(
    env: "DirectRLEnv",
    std: float,
    minimal_height: float,
    target_pos: torch.Tensor = None,
    command_name: str = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Compute reward for reaching and tracking object to goal position.
    
    In Direct workflow, pass target_pos (from env.target_poses).
    In Manager-based workflow, pass command_name (deprecated).
    
    Args:
        env: Environment instance
        std: Standard deviation for tanh scaling
        minimal_height: Minimum height requirement for object
        target_pos: Target position in world frame (for Direct workflow)
        command_name: Command name (for Manager-based workflow, deprecated)
        robot_cfg: Robot configuration
        object_cfg: Object configuration
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    distance = object_goal_distance_real(env, target_pos=target_pos, command_name=command_name, 
                                        robot_cfg=robot_cfg, object_cfg=object_cfg)
    object: RigidObject = env.scene[object_cfg.name]
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_lifted = (object_bottom_z > minimal_height).float()
    tracking_reward = 1 - torch.tanh(distance / std)
    # Strict gating: Only reward tracking if object is successfully lifted
    # This prevents "slidingHack" where agent gets partial reward for sliding object on ground
    return is_lifted * tracking_reward


def object_goal_is_tracked(
    env: "DirectRLEnv",
    threshold: float,
    minimal_height: float,
    target_pos: torch.Tensor = None,
    command_name: str = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has reached and is being tracked at goal position.
    
    In Direct workflow, pass target_pos (from env.target_poses).
    In Manager-based workflow, pass command_name (deprecated).
    
    Args:
        env: Environment instance
        threshold: Distance threshold for "tracked"
        minimal_height: Minimum height requirement
        target_pos: Target position in world frame (for Direct workflow)
        command_name: Command name (for Manager-based workflow, deprecated)
        robot_cfg: Robot configuration
        object_cfg: Object configuration
    
    Returns:
        Binary tracking status tensor of shape (num_envs,)
    """
    distance = object_goal_distance_real(env, target_pos=target_pos, command_name=command_name,
                                        robot_cfg=robot_cfg, object_cfg=object_cfg)
    is_close = (distance < threshold)
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_high_enough = (object_bottom_z > minimal_height)
    is_settled = env.episode_length_buf > 15
    return (is_close & is_high_enough & is_settled).float()


def object_is_reached(
    env: "DirectRLEnv",
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Check if end-effector is within threshold distance of object bounding box."""
    object_ee_distance = object_bbox_ee_distance_real(env, object_cfg, ee_frame_cfg)
    return (object_ee_distance < threshold).float()


def object_bbox_ee_distance_real(
    env: "DirectRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    if not hasattr(env, "object_local_corners"):
        object_entity = env.scene[object_cfg.name]
        ee_frame = env.scene[ee_frame_cfg.name]
        return torch.norm(object_entity.data.root_com_pose_w[:, :3] - ee_frame.data.target_pos_w[..., 0, :], dim=1)

    bbox_extents = torch.max(env.object_local_corners, dim=1)[0]

    object_entity = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos_w = object_entity.data.root_com_pose_w[:, :3]
    obj_quat_w = object_entity.data.root_com_pose_w[:, 3:7]

    ee_pos_local, _ = subtract_frame_transforms(obj_pos_w, obj_quat_w, ee_pos_w, ee_frame.data.target_quat_w[:, 0, :])

    dist_vector = torch.clamp(torch.abs(ee_pos_local) - bbox_extents, min=0.0)
    bbox_distance = torch.norm(dist_vector, dim=-1)
    return bbox_distance


def object_bbox_ee_distance(
    env: "DirectRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    distance = object_bbox_ee_distance_real(env, object_cfg, ee_frame_cfg)
    
    # 雙重解析度混合：一半用於大範圍引導 (std)，一半用於近距離精準微調 (極小 std)
    broad_reward = 1 - torch.tanh(distance / std)
    sharp_reward = 1 - torch.tanh(distance / (std * 0.4))  # 當接近到 std 的 40% 時才給分 (之前是 20%)
    
    return 0.6 * broad_reward + 0.4 * sharp_reward


def check_lifting_success(
    env: "DirectRLEnv",
    threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    floor_height: float = 0.0,
) -> torch.Tensor:
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    lift_height = object_bottom_z - floor_height
    is_success = ((lift_height > threshold) & (env.episode_length_buf > 15)).float()
    return is_success


def check_robust_lifting_success(
    env: "DirectRLEnv",
    min_height: float = 0.05,
    max_dist_to_ee: float = 0.20,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Check if object is lifted AND close to gripper (robust grasp)."""
    # 1. Height check
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_lifted = (object_bottom_z > min_height)
    
    # 2. Distance check (In Hand)
    # Using bounding box distance to EE for accuracy
    dist_to_ee = object_bbox_ee_distance_real(env, object_cfg, ee_frame_cfg) 
    is_close = (dist_to_ee < max_dist_to_ee)
    
    # 3. Combined check
    # settled check included to avoid initial spawning noise
    is_settled = env.episode_length_buf > 10
    
    return (is_lifted & is_close & is_settled).float()


def check_stable_lifting_success(
    env: "DirectRLEnv",
    min_height: float = 0.08,
    min_grasp_quality: float = 0.50,   # [0421 SCOOP FIX] 0.25 → 0.50
    velocity_threshold: float = 0.30,  # [0421 SCOOP FIX] 0.15 → 0.30
    min_distance_to_ee: float = 0.04,
    max_distance_to_ee: float = 0.08,  # [0421 SCOOP FIX] 0.12 → 0.08
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    joint_name: str = "r_joint",
) -> torch.Tensor:
    """
    Enhanced stable lifting detection to prevent penetration-based false positives.
    
    Checks:
    1. Sufficient height (min_height)
    2. Safe distance from EE (no deep penetration)
    3. Low velocity (object is stable, not vibrating)
    4. Adequate grasp quality (gripper is actually holding)
    5. Sufficient episode time (settled)
    
    Args:
        env: Environment instance
        min_height: Minimum height for object bottom (0.08m = 8cm, higher than before)
        min_grasp_quality: Minimum gripper closure (0.25 = 25% closed)
        velocity_threshold: Max vertical velocity of object (m/s)
        min_distance_to_ee: Min distance to prevent penetration (0.04m)
        max_distance_to_ee: Max distance for "in hand" (0.12m, reduced from 0.20)
        
    Returns:
        Boolean tensor indicating stable lift success
    """
    
    # 1. Height check
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_lifted_high = (object_bottom_z > min_height)
    
    # 2. Distance check (prevent penetration)
    dist_to_ee = object_bbox_ee_distance_real(env, object_cfg, ee_frame_cfg)
    is_in_safe_zone = (dist_to_ee <= max_distance_to_ee) #(dist_to_ee >= min_distance_to_ee) & 
    
    # 3. Velocity check (object should be stable, not vibrating/moving fast)
    obj_vel_z = env.scene[object_cfg.name].data.root_lin_vel_w[:, 2]
    is_velocity_low = (torch.abs(obj_vel_z) < velocity_threshold)
    
    # 4. Grasp quality check (gripper actually holding, not just near object)
    robot = env.scene["robot"]
    try:
        joint_indices, _ = robot.find_joints(joint_name)
        gripper_idx = joint_indices[0]
        gripper_pos = robot.data.joint_pos[:, gripper_idx]
        grasp_quality = 1.0 - (gripper_pos / 1.569)  # 0-1, where 1=fully closed
        grasp_quality = torch.clamp(grasp_quality, 0.0, 1.0)
        is_grasped = (grasp_quality > min_grasp_quality)
    except Exception:
        is_grasped = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    
    # 5. Settled check (avoid initial spawning noise)
    is_settled = env.episode_length_buf > 20  # Slightly higher than before
    
    # [0421 SCOOP FIX] Re-enabled is_velocity_low: objects scooped against robot body have high velocity jitter
    # Combined check: ALL conditions must be true
    return (is_lifted_high & is_in_safe_zone & is_grasped & is_settled).float() # & is_velocity_low

def action_rate_l2(env: "DirectRLEnv") -> torch.Tensor:
    """Penalty for large changes in actions (Smoothness/Jitter).
    
    Requires self.previous_actions to be initialized and updated in the environment.
    
    Args:
        env: The environment instance.
        
    Returns:
        Tensor of shape (num_envs,) with L2 norm of action differences.
    """
    if not hasattr(env, "previous_actions") or env.previous_actions is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    return torch.norm(env.actions - env.previous_actions, p=2, dim=-1)


def action_rate_l2_approach(
    env: "DirectRLEnv", 
    threshold: float = 0.05, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")
) -> torch.Tensor:
    """Penalty for large changes in actions (Smoothness/Jitter), but ONLY when close to the object.
    
    Requires self.previous_actions to be initialized and updated in the environment.
    
    Args:
        env: The environment instance.
        threshold: The distance threshold in meters. Penalty is only applied when distance < threshold.
        object_cfg: Object configuration to get position.
        ee_frame_cfg: End-effector configuration to get position.
        
    Returns:
        Tensor of shape (num_envs,) with L2 norm of action differences, masked by proximity.
    """
    if not hasattr(env, "previous_actions") or env.previous_actions is None:
        return torch.zeros(env.num_envs, device=env.device)
        
    # Get distance between EE and object
    ee_frame = env.scene[ee_frame_cfg.name]
    object = env.scene[object_cfg.name]
    
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    object_pos_w = object.data.root_com_pose_w[:, :3]
    distance = torch.norm(ee_pos_w - object_pos_w, dim=1)
    
    # Create mask: 1.0 if close enough, 0.0 otherwise
    is_close = (distance <= threshold).float()
    
    # Calculate action rate penalty and apply mask
    action_penalty = torch.norm(env.actions - env.previous_actions, p=2, dim=-1)
    return action_penalty * is_close

def action_rate_l2_near_goal(
    env: "DirectRLEnv",
    threshold: float = 0.15,      # Increased threshold for smoother transition
    minimal_height: float = 0.08,
    target_pos: torch.Tensor = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for jittering/trembling (action rate) when the object has reached the target area.
    This uses a smooth fall-off to avoid binary 'cliff' effects that cause agents to avoid the goal.
    """
    if not hasattr(env, "previous_actions") or env.previous_actions is None:
        return torch.zeros(env.num_envs, device=env.device)
        
    # Check if object is tracked (lifted and near goal)
    from .rewards import object_goal_distance_real, _get_object_bottom_z
    distance = object_goal_distance_real(env, target_pos=target_pos, robot_cfg=SceneEntityCfg("robot"), object_cfg=object_cfg)
    
    # Smooth mask: 1.0 at goal, falls to 0.0 at ~2*threshold
    # Using tanh for a smooth "bell" shape
    is_close_smooth = 1.0 - torch.tanh(distance / threshold)
    
    object_bottom_z = _get_object_bottom_z(env, object_cfg)
    is_high_enough = (object_bottom_z > minimal_height).float()
    
    # Calculate action rate penalty and mask it smoothly
    action_penalty = torch.norm(env.actions - env.previous_actions, p=2, dim=-1)
    return action_penalty * is_close_smooth * is_high_enough



def joint_vel_l2(
    env: "DirectRLEnv", 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """Penalty for joint velocities exceeding a safe threshold.
    
    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot asset.
        threshold: Safe velocity limit. Only velocities above this are penalized.
        
    Returns:
        Tensor of shape (num_envs,) with excess velocity penalty.
    """
    asset = env.scene[asset_cfg.name]
    vel_norm = torch.norm(asset.data.joint_vel, p=2, dim=-1)
    
    # 核心邏輯：只對超過 threshold 的部分進行懲罰
    # 這樣在低速移動時不會有扣分，鼓勵 Agent 緩慢動作而不是停在原地
    excess = torch.clamp(vel_norm - threshold, min=0.0)
    
    # 用於防範速度爆炸：如果真的爆掉了，我們給予一個飽和的極大負分，但確保數值穩定 (clamp)
    return torch.clamp(excess, max=20.0)


def object_in_view_reward(
    env: "DirectRLEnv",
    camera_sensor_name: str = "camera_low",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    focus_exponent: float = 2.0,
) -> torch.Tensor:
    """
    Reward for keeping the object within the camera's field of view and centered.
    
    The reward is calculated based on the geometric projection of the object's position
    onto the camera's image plane.
    
    Args:
        env: The environment instance.
        camera_sensor_name: Name of the camera sensor in env.scene.
        object_cfg: Scene entity configuration for the object.
        focus_exponent: Exponent for the focus term (higher = stricter centering).
        
    Returns:
        Reward tensor (num_envs,) in range [0, 1].
        1.0 = Centered in view.
        0.0 = Out of view.
    """
    # 1. Get Sensor and Object Data
    camera = env.scene[camera_sensor_name]
    object_entity = env.scene[object_cfg.name]
    
    # Object Position (World Frame)
    obj_pos_w = object_entity.data.root_pos_w  # (N, 3)
    
    # Camera Pose (World Frame)
    # TiledCamera data: pos_w (N, 3), quat_w_world (N, 4)
    cam_pos_w = camera.data.pos_w
    cam_quat_w = camera.data.quat_w_world
    
    # Camera Intrinsics
    # (N, 3, 3)
    K = camera.data.intrinsic_matrices
    
    # Image Dimensions
    # (H, W) or (H, W, 1) depending on implementation
    height, width = camera.data.image_shape[:2]
    
    # 2. Transform World -> Camera Frame
    # P_c = R_inv * (P_w - T_c)
    # R_inv is conjugate/inverse of cam_quat_w
    from isaaclab.utils.math import quat_inv, quat_apply
    
    cam_quat_inv = quat_inv(cam_quat_w)
    vec_w = obj_pos_w - cam_pos_w
    obj_pos_c = quat_apply(cam_quat_inv, vec_w) # (N, 3) (x, y, z) in cam frame
    
    # 3. Project to Image Plane
    # u = fx * (x/z) + cx
    # v = fy * (y/z) + cy
    # Manual projection
    
    # Expand for batch matmul: (N, 3, 1)
    obj_pos_c_expanded = obj_pos_c.unsqueeze(-1) 
    projected = torch.bmm(K, obj_pos_c_expanded).squeeze(-1) # (N, 3)
    
    # Normalize by Z (depth)
    z = projected[:, 2]
    # Guard against division by zero
    z_safe = torch.where(torch.abs(z) < 1e-6, torch.ones_like(z) * 1e-6, z)
    
    u = projected[:, 0] / z_safe
    v = projected[:, 1] / z_safe
    
    # 4. Check Visibility
    # Condition 1: In front of camera (z > 0.05)
    in_front = (z > 0.05)
    
    # Condition 2: Within image bounds [0, W] and [0, H]
    in_bounds_u = (u >= 0) & (u < width)
    in_bounds_v = (v >= 0) & (v < height)
    
    is_visible = in_front & in_bounds_u & in_bounds_v # (N,)

    
    # 5. Calculate Focus Reward (Centering)
    center_u = width / 2.0
    center_v = height / 2.0
    
    # Normalized distance from center (0 = center, 1 = edge)
    # Norm by half-dim
    dist_u = (u - center_u) / (width / 2.0)
    dist_v = (v - center_v) / (height / 2.0)
    
    # Euclidean distance from center in normalized coords range [0, sqrt(2)]
    # dist = sqrt(du^2 + dv^2)
    dist_norm_sq = dist_u**2 + dist_v**2
    dist_norm = torch.sqrt(dist_norm_sq)
    
    # Max possible distance is sqrt(1^2 + 1^2) = 1.414 (corner)
    # We want 1.0 at center, 0.0 at corners
    # Normalize to [0, 1] range: dist_norm / 1.414
    dist_scaled = torch.clamp(dist_norm / 1.4142, 0.0, 1.0)
    
    # Invert so 0 dist -> 1 reward
    # Use exponent to sharpen the peak around center
    focus_score = (1.0 - dist_scaled)
    focus_score = torch.clamp(focus_score, 0.0, 1.0)
    focus_score = torch.pow(focus_score, focus_exponent)
    
    # Mask out invisible objects (reward = 0.0)
    total_reward = focus_score * is_visible.float()
    
    return total_reward


def grasp_pose_alignment(
    env: "DirectRLEnv",
    pos_std: float = 0.05,
    rot_weight: float = 0.3,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward for aligning EE with nearest predicted grasp pose (0403 CGN).

    Position component: 1 - tanh(||ee_pos - grasp_pos|| / pos_std)
    Orientation component: max(0, dot(ee_approach_dir, grasp_approach_dir))
      where approach direction = +Z axis of the respective frame.

    Combined: (1 - rot_weight) * pos_reward + rot_weight * rot_reward

    Returns 0.0 for environments without valid grasps (neutral — doesn't hurt training).

    Args:
        env: Environment instance (must have nearest_grasp_pos_w, nearest_grasp_quat_w,
             has_valid_grasps attributes from 0403 env).
        pos_std: Standard deviation for tanh scaling of position error (metres).
        rot_weight: Blend weight of orientation component vs position (0–1).
        ee_frame_cfg: Scene entity config for end-effector frame.

    Returns:
        Reward tensor (num_envs,) in range [0, 1].
    """
    from isaaclab.utils.math import quat_apply as _quat_apply

    # Guard: if the env doesn't have CGN data, return zeros
    if not hasattr(env, "nearest_grasp_pos_w") or not hasattr(env, "has_valid_grasps"):
        return torch.zeros(env.num_envs, device=env.device)

    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]    # (B, 3)
    ee_quat_w = env.scene[ee_frame_cfg.name].data.target_quat_w[..., 0, :]  # (B, 4)

    # ── Position alignment ────────────────────────────────────────────
    pos_dist = torch.norm(ee_pos_w - env.nearest_grasp_pos_w, dim=-1)  # (B,)
    pos_reward = 1.0 - torch.tanh(pos_dist / pos_std)

    # ── Orientation alignment ─────────────────────────────────────────
    # Compare +Z axes (approach direction for the JetRover gripper)
    z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3)
    ee_z = _quat_apply(ee_quat_w, z_axis)                      # (B, 3)
    grasp_z = _quat_apply(env.nearest_grasp_quat_w, z_axis)    # (B, 3)
    # Dot product: 1 = aligned, 0 = perpendicular, -1 = opposite
    rot_reward = torch.clamp(torch.sum(ee_z * grasp_z, dim=-1), min=0.0)  # (B,)

    # ── Combine ───────────────────────────────────────────────────────
    combined = (1.0 - rot_weight) * pos_reward + rot_weight * rot_reward

    # Mask invalid environments
    return combined * env.has_valid_grasps.float()