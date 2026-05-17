from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def object_reached_goal(
    env: "DirectRLEnv",
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination when object reaches goal position."""
    try:
        robot: RigidObject = env.scene[robot_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_b = command[:, :3]
        des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
        distance = torch.norm(des_pos_w - object.data.root_com_pose_w[:, :3], dim=1)
        return distance < threshold
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def object_out_of_workspace_rect(
    env: "DirectRLEnv",
    rect_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination when object exceeds rectangular workspace relative to robot.
    
    Args:
        env: The environment.
        rect_range: Dictionary with keys "x" and "y" containing (min, max) tuples.
        asset_cfg: The object configuration.
        
    Returns:
        Tensor of shape (num_envs,) indicating if object is out of bounds.
    """
    try:
        # 1. Get object and robot world positions
        object_pos_w = env.scene[asset_cfg.name].data.root_com_pos_w
        robot_pos_w = env.scene["robot"].data.root_pos_w
        
        # 2. Calculate relative position (Vector from Robot Base to Object)
        # This ensures the check works "env to env" (relative to each robot's base)
        relative_pos = object_pos_w - robot_pos_w
        
        # 3. Check X bounds
        x_min, x_max = rect_range["x"]
        out_of_x = (relative_pos[:, 0] < x_min) | (relative_pos[:, 0] > x_max)
        
        # 4. Check Y bounds
        y_min, y_max = rect_range["y"]
        out_of_y = (relative_pos[:, 1] < y_min) | (relative_pos[:, 1] > y_max)
        
        # 5. Combined check (Out of X OR Out of Y)
        return out_of_x | out_of_y
        
    except Exception:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
