from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def object_position_in_robot_root_frame(
    env: "DirectRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_com_pose_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def object_bbox_corners(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    """
    Calculate 8 bounding-box corners for the object (world frame) and return flattened (num_envs, 24).
    """
    if not hasattr(env, "object_local_corners"):
        return torch.zeros((env.num_envs, 24), device=env.device)

    # object pose
    object_body = env.scene[object_cfg.name]
    pos = object_body.data.root_com_pose_w[:, :3]
    quat = object_body.data.root_com_pose_w[:, 3:7]

    batch_size = env.num_envs
    flat_corners = env.object_local_corners.view(batch_size * 8, 3)
    flat_quat = quat.repeat_interleave(8, dim=0)

    rotated_corners = quat_apply(flat_quat, flat_corners)
    flat_pos = pos.repeat_interleave(8, dim=0)
    world_corners = rotated_corners + flat_pos

    return world_corners.view(batch_size, 24)


def object_bbox_corners_relative(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")):
    world_corners_flat = object_bbox_corners(env, object_cfg).view(env.num_envs * 8, 3)

    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w.repeat_interleave(8, dim=0)
    robot_quat = robot.data.root_quat_w.repeat_interleave(8, dim=0)

    local_corners_flat, _ = subtract_frame_transforms(robot_pos, robot_quat, world_corners_flat, robot_quat)

    return local_corners_flat.view(env.num_envs, 24)
