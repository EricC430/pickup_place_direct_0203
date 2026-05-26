# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import itertools
import random
import collections
from collections import deque
from collections.abc import Sequence

# # Fallback for cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR during XFormPrim initialization
# try:
#     if hasattr(torch.backends.cuda, "preferred_linalg_library"):
#         torch.backends.cuda.preferred_linalg_library("magma")
# except Exception:
#     pass

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import FrameTransformer, ContactSensor, TiledCamera
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg
from isaaclab.sim.spawners.shapes import SphereCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg

from .pickup_place_direct_0510_env_cfg import PickupPlaceDirect0510EnvCfg, SELECTED_OBJECT_IDS, OBJECT_FRICTION_MAP
from .mdp import observations as mdp_obs
from .mdp import rewards as mdp_rewards
from .mdp import terminations as mdp_term
from .mdp import curriculum as mdp_curriculum


class PickupPlaceDirect0510Env(DirectRLEnv):
    """Direct RL environment for pickup-place task with JetRover (Task-Space Delta IK)."""

    cfg: PickupPlaceDirect0510EnvCfg

    def __init__(self, cfg: PickupPlaceDirect0510EnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize object randomization info BEFORE super().__init__
        # (because super().__init__ calls _setup_scene which needs these attributes)
        self._object_ids = SELECTED_OBJECT_IDS
        self._num_objects = len(self._object_ids)

        # Call parent init (will call _setup_scene internally)
        super().__init__(cfg, render_mode, **kwargs)
        # Find all controlled joints
        self._arm_joint_indices, self._arm_joint_names = self.robot.find_joints(["joint1", "joint2", "joint3", "joint4", "joint5"])
        self._gripper_joint_idx, _ = self.robot.find_joints("r_joint")

        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel

        # [NEW] Setup Differential IK Controller
        ee_ids, _ = self.robot.find_bodies("gripper_link")
        if len(ee_ids) > 0:
            self.ee_body_idx = ee_ids[0]
        else:
            self.ee_body_idx = -1

        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method=self.cfg.action_cfg.get("ik_method", "dls"),
        )
        self.ik_controller = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # Initialize bounding box corners for all objects
        self._initialize_object_local_corners()

        # Curriculum tracking - Initialize with all reward term weights
        self._curriculum_term_weights = {
            "lifting_object": 0.0,
            "lifting_object_velocity": 0.0,
            "close_reward": 0.0,
            "object_goal_tracking": 0.0,
            "object_goal_tracking_fine_grained": 0.0,
            "action_rate": 0.0,
            "joint_vel": 0.0,
        }

        # [Manual Override] Initialize weights from config if specified (for resuming training)
        if hasattr(self.cfg, "curriculum_starting_weights"):
            for key, val in self.cfg.curriculum_starting_weights.items():
                self._curriculum_term_weights[key] = val

        # Buffers for Anytime Consecutive Success Check
        self.consecutive_success_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_max_success = torch.zeros(self.num_envs, device=self.device) # Latched success status
        
        # Configuration for success detection
        self.consecutive_success_threshold = self.cfg.success_criteria["consecutive_success_threshold"]
        self.initial_ignore_steps = self.cfg.success_criteria["initial_ignore_steps"]

        # Buffer for curriculum tracking
        self.current_rolling_success = 0.0
        # ISSUE FIX: Increase buffer size to num_envs to prevent flushing data when all envs reset
        self.success_history = deque(maxlen=self.num_envs)
        
        # [NEW] Object ID Buffer for data collection & playback metadata
        self.object_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._detect_object_id()
        
        # Accumulator for Episode Sum Rewards (Total Return)
        # Dictionary mapping reward name -> tensor of current episode accumulated reward
        self.episode_sums = {
            "reward_reaching": torch.zeros(self.num_envs, device=self.device),
            "reward_lifting": torch.zeros(self.num_envs, device=self.device),
            "reward_lifting_vel": torch.zeros(self.num_envs, device=self.device),
            "reward_close": torch.zeros(self.num_envs, device=self.device),
            "reward_goal": torch.zeros(self.num_envs, device=self.device),
            "reward_goal_fine": torch.zeros(self.num_envs, device=self.device),
            "reward_action_rate": torch.zeros(self.num_envs, device=self.device),
            "reward_joint_vel": torch.zeros(self.num_envs, device=self.device),
            "reward_torques": torch.zeros(self.num_envs, device=self.device),
        }
        
        # Buffer to hold finished episode stats to be logged in the next step
        self._pending_episode_log_buffer = collections.defaultdict(list)

        # Pending reward logs for TensorBoard (used by the wrapper)
        self._pending_reward_logs = {}
        
        # [New] Population-Based Success Metric Buffer
        # Stores the latest episode outcome for each environment
        self.env_goal_success = torch.zeros(self.num_envs, device=self.device)
        self.env_lifting_success = torch.zeros(self.num_envs, device=self.device)
        self.env_reaching_success = torch.zeros(self.num_envs, device=self.device)
        
        print(f"\n[Info] Environment initialized with {self._num_objects} possible object types")

        # Command generation
        self._generate_commands()

        # Debug markers
        self._setup_markers()
        
        # Setup bounding box corner visualization (ISSUE FIX #3)
        self._setup_bbox_visualizer()
        
        # Initialize random seeds for arm initial positions
        self._init_arm_randomization()

        # Setup gripper friction (one-time) and prepare friction API cache
        self._setup_friction_materials()

    def _setup_scene(self):
        # Add robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Add end-effector frame transformer
        self.ee_frame = FrameTransformer(self.cfg.ee_frame_cfg)

        # Add fingertip contact sensors
        self.left_finger_force = ContactSensor(self.cfg.left_finger_force)
        self.right_finger_force = ContactSensor(self.cfg.right_finger_force)

        # Clone environments
        # using copy_from_source=True to ensure environments are independent copies (not instances)
        # this is required for heterogeneous object spawning
        self.scene.clone_environments(copy_from_source=True)

        # Add object wrapper (picks up the spawned prims via regex)
        self.object = RigidObject(self.cfg.object_cfg)

        # Filter collisions for CPU
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Register to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["ee_frame"] = self.ee_frame
        self.scene.sensors["left_finger_force"] = self.left_finger_force
        self.scene.sensors["right_finger_force"] = self.right_finger_force

        # Spawn wrist camera if configured
        if hasattr(self.cfg, "wrist_camera_cfg") and self.cfg.wrist_camera_cfg is not None:
            self.wrist_camera = TiledCamera(self.cfg.wrist_camera_cfg)
            self.scene.sensors["wrist_camera"] = self.wrist_camera
            print(f"[Env0510] Wrist Camera added: {self.cfg.wrist_camera_cfg.prim_path} (Tiled)")

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _setup_markers(self):
        """Setup visualization markers for debugging."""
        # Red marker for object center of mass
        com_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CoM_Marker",
            markers={
                "default": SphereCfg(
                    radius=0.015,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
        self.com_marker = VisualizationMarkers(com_marker_cfg)

        # Target marker (Green)
        target_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TargetMarker",
            markers={
                "default": SphereCfg(
                    radius=0.015,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            },
        )
        self.target_marker = VisualizationMarkers(target_marker_cfg)
    
    def _setup_bbox_visualizer(self):
        """Setup bounding box corner visualization (ISSUE FIX #3).
        
        Creates orange sphere markers at the 8 corners of each object's bounding box.
        This helps debug object orientation and size during episodes.
        """
        bbox_marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/BBoxCorners",
            markers={
                "corner": SphereCfg(
                    radius=0.008,
                    visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),  # Orange for bbox corners
                )
            },
        )
        self.bbox_visualizer = VisualizationMarkers(bbox_marker_cfg)    
    def _init_arm_randomization(self):
        """Initialize buffers for arm joint randomization at reset."""
        if not self.cfg.randomize_arm_init:
            return
        
        # Pre-allocate random arm target buffer
        self.random_arm_targets = torch.zeros(
            (self.num_envs, 5), device=self.device, dtype=torch.float32
        )
    def _update_object_local_corners(self, env_ids: torch.Tensor | None = None):
        """Dynamically compute bounding box corners for specified environments.
        
        This method updates the local corner cache for the specified environments.
        If env_ids is None, updates all environments. This should be called every
        reset to ensure corners match the current object pose and shape.
        """
        import isaaclab.sim as sim_utils_local
        from pxr import UsdGeom, Usd

        try:
            stage = sim_utils_local.get_current_stage()
        except Exception:
            return

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        env_ids_list = env_ids.cpu().tolist()
        
        # Initialize tensor if not exist yet (for first call)
        if not hasattr(self, 'object_local_corners') or self.object_local_corners.shape[0] != self.num_envs:
            # Create full tensor with default values
            base_corners = torch.tensor(
                list(itertools.product([1, -1], repeat=3)), dtype=torch.float32, device=self.device
            )
            default_half_extents = torch.ones((self.num_envs, 3), device=self.device) * 0.05
            local_corners = base_corners.unsqueeze(0) * default_half_extents.unsqueeze(1) * 0.6
            self.object_local_corners = local_corners

        # Now update only the specified environments
        for idx, env_id in enumerate(env_ids_list):
            prim_path = f"/World/envs/env_{env_id}/Object"
            prim = stage.GetPrimAtPath(prim_path)
            
            half_extents = torch.ones(3, device=self.device) * 0.05
            centers = torch.zeros(3, device=self.device)
            
            if prim.IsValid():
                bound = bbox_cache.ComputeUntransformedBound(prim)
                box_range = bound.GetRange()
                min_pt = box_range.GetMin()
                max_pt = box_range.GetMax()

                hx = (max_pt[0] - min_pt[0]) / 2.0
                hy = (max_pt[1] - min_pt[1]) / 2.0
                hz = (max_pt[2] - min_pt[2]) / 2.0
                half_extents = torch.tensor([hx, hy, hz], dtype=torch.float32, device=self.device)
                centers = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

            config_scale = torch.tensor(self.cfg.object_scale, device=self.device)
            half_extents = half_extents * config_scale

            base_corners = torch.tensor(
                list(itertools.product([1, -1], repeat=3)), dtype=torch.float32, device=self.device
            )
            local_corners = base_corners * half_extents + centers
            self.object_local_corners[env_id] = local_corners
    
    def _initialize_object_local_corners(self):
        """Initialize object local corners for all environments."""
        self._update_object_local_corners()

    def _update_debug_vis(self):
        """Update visualization markers for debugging.
        
        This method checks the configuration settings and updates the markers accordingly.
        It is called at every step and reset to ensure dynamic visualization.
        """
        # Only visualize if rendering is enabled (PARTIAL_RENDERING or better)
        if self.sim.render_mode < self.sim.RenderMode.PARTIAL_RENDERING:
            return

        # Check master toggle
        if not self.cfg.debug_vis:
            return

        settings = self.cfg.debug_vis_settings

        # Target position marker (green sphere) - shows desired goal
        if settings.get("target", True) and hasattr(self, "target_marker"):
            try:
                self.target_marker.visualize(self.target_poses + self.scene.env_origins)
            except Exception:
                pass

        # Center of Mass marker (red sphere) - shows actual object CoM position
        if settings.get("com", True) and hasattr(self, "com_marker"):
            try:
                object_entity = self.scene["object"]
                # Get CoM position in world frame
                com_pos_w = object_entity.data.root_com_pose_w[:, :3]
                self.com_marker.visualize(translations=com_pos_w)
            except Exception:
                pass

        # Bounding box corners (orange spheres) - shows bbox for current object
        if settings.get("bbox", True) and hasattr(self, "bbox_visualizer") and hasattr(self, "object_local_corners"):
            try:
                object_entity = self.scene["object"]
                
                # Get object poses for ALL environments (since we update every step)
                obj_pos_w = object_entity.data.root_com_pose_w[:, :3]
                obj_rot_w = object_entity.data.root_quat_w[:, :]
                
                # Transform local bbox corners to world frame
                # self.object_local_corners: (num_envs, 8, 3)
                
                local_corners = self.object_local_corners # (num_envs, 8, 3)
                
                # Expand rotation for broadcasting: (num_envs, 4) -> (num_envs, 8, 4)
                # Explicit expansion to ensure compatibility with quat_apply
                rot_expanded = obj_rot_w.unsqueeze(1).expand(-1, 8, -1)
                
                # Apply rotation: q * v * q^-1
                corners_rot = quat_apply(rot_expanded, local_corners)
                
                # Add position offset: (num_envs, 8, 3) + (num_envs, 1, 3)
                corners_world = corners_rot + obj_pos_w.unsqueeze(1)
                
                # Flatten: (num_envs * 8, 3)
                world_corners_all = corners_world.view(-1, 3)
                
                self.bbox_visualizer.visualize(translations=world_corners_all)
            except Exception as e:
                print(f"[Error] BBox visualization failed: {e}")

    def _reset_scene_elements(self, env_ids: Sequence[int]):
        """Reset all scene elements to their initial state.
        
        This is the Direct workflow equivalent of the reset_scene_to_default event
        in the Manager-Based architecture. It ensures all dynamic scene components
        are properly reset when environments are reset.
        
        Args:
            env_ids: Indices of environments to reset.
        """
        # Reset end-effector frame transformer (ensures correct frame tracking)
        if hasattr(self, "ee_frame"):
            try:
                self.ee_frame.update()
            except Exception:
                pass  # Frame transformer may need scene updates, continue if fails
        
        # Dynamically update object bounding box corners for current positions (ISSUE FIX #1)
        # REMOVED: self._update_object_local_corners(env_ids)
        # Object local bounding boxes are static and pre-computed in __init__

        # Update visualization markers (ensures visual debugging aids are in place)
        self._update_debug_vis()

    def _generate_commands(self):
        """Generate target poses for curriculum.
        
        This is the Direct workflow equivalent of CommandManager in Manager-Based,
        which generates random target positions for the object during each episode.
        The target_poses are used as observation input for the policy network,
        allowing the agent to learn goal-directed behavior.
        """
        self.target_poses = torch.zeros((self.num_envs, 3), device=self.device)
        self._sample_target_poses()

    def _sample_target_poses(self, env_ids: torch.Tensor | None = None):
        """Sample random target poses in the robot's workspace.
        
        Configuration values (from env_cfg):
        - target_pos_range["x"]: (0.05, 0.25)
        - target_pos_range["y"]: (-0.2, 0.2)
        - target_pos_range["z"]: (0.3, 0.5)
        
        These define the range of valid target positions for the object
        in the robot's base frame.
        
        Args:
            env_ids: Indices of environments to sample for. If None, samples for all environments.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        cfg = self.cfg
        x_range = cfg.target_pos_range["x"]
        y_range = cfg.target_pos_range["y"]
        z_range = cfg.target_pos_range["z"]

        # Uniformly sample target positions within the specified ranges for the given env_ids only
        self.target_poses[env_ids, 0] = torch.rand(len(env_ids), device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
        self.target_poses[env_ids, 1] = torch.rand(len(env_ids), device=self.device) * (y_range[1] - y_range[0]) + y_range[0]
        self.target_poses[env_ids, 2] = torch.rand(len(env_ids), device=self.device) * (z_range[1] - z_range[0]) + z_range[0]
    
    def _randomize_arm_init_positions(self, env_ids: torch.Tensor | None = None):
        """Generate random arm initial joint targets to vary viewing angles.
        
        This method randomizes the arm's initial position at episode reset,
        allowing the camera to view the object from different angles while
        keeping the object approximately in the center of the camera frame.
        
        The joint targets are sampled from configured ranges and ensure
        the robot maintains a functional grasp configuration.
        
        Args:
            env_ids: Indices of environments to randomize. If None, randomizes all.
        """
        if not self.cfg.randomize_arm_init or not hasattr(self, 'random_arm_targets'):
            return
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        cfg = self.cfg
        
        # Sample random arm joint targets within configured ranges
        joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5"]
        for i, joint_name in enumerate(joint_names):
            joint_range = cfg.arm_init_offset_range[joint_name]
            self.random_arm_targets[env_ids, i] = (
                torch.rand(len(env_ids), device=self.device) * 
                (joint_range[1] - joint_range[0]) + joint_range[0]
            )
        # Note: We only sample here. Application is handled in _reset_idx for consistency.
    
    def _randomize_object_yaw(self, env_ids: torch.Tensor | None = None):
        """Randomize object orientation (yaw/horizontal rotation) at reset.
        
        This increases training diversity by varying the object's initial orientation.
        The object's position remains unchanged, only rotation around Z-axis is modified.
        
        Args:
            env_ids: Indices of environments to randomize. If None, randomizes all.
        """
        if not self.cfg.randomize_object_yaw:
            return
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # Sample random yaw angles (horizontal rotation)
        yaw_range = self.cfg.object_yaw_range
        random_yaws = (
            torch.rand(len(env_ids), device=self.device) * 
            (yaw_range[1] - yaw_range[0]) + yaw_range[0]
        )
        
        # Convert yaw to quaternion (rotation around Z-axis)
        # Formula: quat = [cos(yaw/2), 0, 0, sin(yaw/2)]
        half_yaw = random_yaws / 2.0
        cos_half = torch.cos(half_yaw)
        sin_half = torch.sin(half_yaw)
        
        # Create quaternions (w, x, y, z) for Z-axis rotation
        random_quats = torch.stack([
            cos_half,           # w
            torch.zeros_like(cos_half),  # x
            torch.zeros_like(cos_half),  # y
            sin_half            # z
        ], dim=1)  # Shape: (len(env_ids), 4)
        
        # Update object orientation
        self.object.data.root_quat_w[env_ids] = random_quats

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        """Apply action via Differential IK and Binary Gripper."""
        # actions shape: (num_envs, 7) -> [dx, dy, dz, drx, dry, drz, gripper]
        
        position_scale = self.cfg.action_cfg.get("position_scale", 0.005)
        rotation_scale = self.cfg.action_cfg.get("rotation_scale", 0.05)
        
        # 1. Parsing Position and Rotation commands
        ik_commands = torch.zeros((self.num_envs, 6), device=self.device)
        ik_commands[:, :3] = self.actions[:, :3] * position_scale
        ik_commands[:, 3:] = self.actions[:, 3:6] * rotation_scale
        
        # Current ee pose for relative IK
        if hasattr(self, 'ee_frame'):
            ee_pos_w = self.ee_frame.data.target_pos_w[..., 0, :]
            ee_quat_w = self.ee_frame.data.target_quat_w[..., 0, :]
        else:
            ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            ee_quat_w = torch.zeros((self.num_envs, 4), device=self.device)
            ee_quat_w[:, 0] = 1.0
            
        self.ik_controller.set_command(ik_commands, ee_pos=ee_pos_w, ee_quat=ee_quat_w)
        
        # 2. Compute IK Jacobian
        ee_jacobians = self.scene["robot"].root_physx_view.get_jacobians()[:, self.ee_body_idx, :, :]
        # Extract only the columns corresponding to the arm joints
        arm_jacobians = ee_jacobians[:, :, self._arm_joint_indices]
        
        # 3. Compute target joint positions
        # Provide the current TARGET arm joint positions to prevent gravity droop
        arm_joint_pos = self.scene["robot"].data.joint_pos_target[:, self._arm_joint_indices]
        joint_pos_desired = self.ik_controller.compute(
            ee_pos_w,
            ee_quat_w,
            arm_jacobians,
            arm_joint_pos
        )
        
        # 4. Binary Gripper Control
        gripper_action = self.actions[:, 6]
        # > 0 means Open (0.0), <= 0 means Close (1.57)
        gripper_target = torch.where(gripper_action > 0.0, torch.zeros_like(gripper_action), torch.full_like(gripper_action, 1.57))
        
        # 🔧 Sim2Real Safety Fix: Clamp to joint limits (arm: [-2.09, 2.09], gripper: [0.0, 1.57])
        joint_pos_desired = torch.clamp(joint_pos_desired, -2.09, 2.09)
        gripper_target = torch.clamp(gripper_target, 0.0, 1.57)

        # 5. Combine and apply
        # joint_pos_desired now strictly contains the 5 arm joints
        targets = torch.cat([joint_pos_desired, gripper_target.unsqueeze(1)], dim=1)
        self.robot.set_joint_position_target(targets, joint_ids=list(self._arm_joint_indices) + list(self._gripper_joint_idx))

    def _get_observations(self) -> dict:
        """Collect observations (state-based, no vision)."""
        # Joint positions/velocities (Relative to default)
        if hasattr(self.cfg, "use_46_dim_obs") and self.cfg.use_46_dim_obs:
            # Previous incorrect version: missing gripper pos and vel (5 + 5 = 10 instead of 12)
            joint_indices = list(self._arm_joint_indices)
        else:
            joint_indices = list(self._arm_joint_indices) + list(self._gripper_joint_idx)
            
        jpos = self.joint_pos[:, joint_indices] - self.robot.data.default_joint_pos[:, joint_indices]
        jvel = self.joint_vel[:, joint_indices] - self.robot.data.default_joint_vel[:, joint_indices]

        # Object position in robot frame
        obj_pos = mdp_obs.object_position_in_robot_root_frame(self, object_cfg=SceneEntityCfg("object"))

        # Bounding box corners
        bbox = mdp_obs.object_bbox_corners_relative(self, object_cfg=SceneEntityCfg("object"))

        # Target position (Fix: Convert from Env Frame -> Robot Frame)
        # 1. Env Frame -> World Frame
        target_pos_w = self.target_poses + self.scene.env_origins
        # 2. World Frame -> Robot Frame (Ego-centric)
        target_in_robot, _ = subtract_frame_transforms(
            self.scene["robot"].data.root_pos_w, 
            self.scene["robot"].data.root_quat_w, 
            target_pos_w
        )
        target = target_in_robot

        # Add Noise (if corruption enabled, typically checked via self.cfg.observation_noise_scale > 0)
        # Using simple additive Gaussian noise logic similar to ManagerBasedEnv defaults
        if self.cfg.observation_noise_scale > 0.0:
            noise_scale = self.cfg.observation_noise_scale
            jpos += torch.randn_like(jpos) * 0.0001 * noise_scale
            jvel += torch.randn_like(jvel) * 0.001 * noise_scale
            obj_pos += torch.randn_like(obj_pos) * 0.001 * noise_scale
            # bbox is derived from object pos/rot, technically should add noise to source or result.
            # Here adding to result for simplicity matching 'object_bbox' noise
            bbox += torch.randn_like(bbox) * 0.001 * noise_scale
            # target typically noiseless in obs, but 'target_object_position' might have noise. 
            # Keeping target clean for now as it's a command.

        # NOTE: Target marker visualization is now updated only in _reset_scene_elements
        # This avoids excessive position updates every step and makes debugging clearer
        # by showing the target only at episode reset time

        # Combine all
        obs = torch.cat(
            [
                jpos.view(self.num_envs, -1),
                jvel.view(self.num_envs, -1),
                obj_pos,
                bbox,
                target,
                self.actions,
            ],
            dim=1,
        )

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards using curriculum-weighted terms."""
        rewards_dict = {}

        # Reaching reward
        rewards_dict["reaching"] = mdp_rewards.object_bbox_ee_distance(
            self, std=self.cfg.reward_settings["reaching_std"], object_cfg=SceneEntityCfg("object")
        )

        # Lifting reward
        rewards_dict["lifting"] = mdp_rewards.object_is_lifted(self, minimal_height=self.cfg.reward_settings["lifting_min_height"], object_cfg=SceneEntityCfg("object"))

        # Lifting velocity
        rewards_dict["lifting_vel"] = mdp_rewards.lifting_velocity_reward(self, object_cfg=SceneEntityCfg("object"))

        # Gripper closing phase
        rewards_dict["close"] = mdp_rewards.gripper_open_close_phases(self, joint_name="r_joint", close_threshold=self.cfg.reward_settings["close_threshold"])

        # Goal tracking (DIRECT WORKFLOW FIX: Use target_poses instead of command_manager)
        # CRITICAL FIX: Add env_origins to target_poses to convert to world frame
        target_pos_w = self.target_poses + self.scene.env_origins
        rewards_dict["goal"] = mdp_rewards.object_goal_distance(
            self, std=self.cfg.reward_settings["goal_std"], minimal_height=self.cfg.reward_settings["goal_min_height"], target_pos=target_pos_w, object_cfg=SceneEntityCfg("object")
        )
        rewards_dict["goal_fine"] = mdp_rewards.object_goal_distance(
            self, std=self.cfg.reward_settings["goal_fine_std"], minimal_height=self.cfg.reward_settings["goal_fine_min_height"], target_pos=target_pos_w, object_cfg=SceneEntityCfg("object")
        )

        # Action rate penalty
        rewards_dict["action_rate"] = mdp_rewards.action_rate_l2(self)

        # Joint velocity penalty
        rewards_dict["joint_vel"] = mdp_rewards.joint_vel_l2(self, asset_cfg=SceneEntityCfg("robot"))

        # Torques penalty
        rewards_dict["torques"] = torch.sum(torch.square(self.robot.data.applied_torque), dim=1)

        # Get curriculum weights (default scale from cfg)
        w_reach = self.cfg.rew_scale_reach
        w_lift = self._curriculum_term_weights.get("lifting_object", self.cfg.rew_scale_lift)
        w_lift_vel = self._curriculum_term_weights.get("lifting_object_velocity", self.cfg.rew_scale_lift_vel)
        w_close = self._curriculum_term_weights.get("close_reward", self.cfg.rew_scale_close)
        w_goal = self._curriculum_term_weights.get("object_goal_tracking", self.cfg.rew_scale_goal)
        w_goal_fine = self._curriculum_term_weights.get("object_goal_tracking_fine_grained", self.cfg.rew_scale_goal_fine)
        w_action = self._curriculum_term_weights.get("action_rate", self.cfg.rew_scale_action)
        w_joint = self._curriculum_term_weights.get("joint_vel", self.cfg.rew_scale_joint_vel)
        w_torques = self._curriculum_term_weights.get("torques", getattr(self.cfg, "rew_scale_torques", 0.0))

        # Compute total reward
        total_reward = (
            w_reach * rewards_dict["reaching"]
            + w_lift * rewards_dict["lifting"]
            + w_lift_vel * rewards_dict["lifting_vel"]
            + w_close * rewards_dict["close"]
            + w_goal * rewards_dict["goal"]
            + w_goal_fine * rewards_dict["goal_fine"]
            + w_action * rewards_dict["action_rate"]
            + w_joint * rewards_dict["joint_vel"]
            + w_torques * rewards_dict["torques"]
        )
        
        # Accumulate Episode Sums (ISSUE FIX #6)
        # Instead of logging step averages, we accumulate total rewards and log them on episode completion
        if w_reach > 0.0:
            self.episode_sums["reward_reaching"] += w_reach * rewards_dict["reaching"]
        if w_lift > 0.0:
            self.episode_sums["reward_lifting"] += w_lift * rewards_dict["lifting"]
        if w_lift_vel > 0.0:
            self.episode_sums["reward_lifting_vel"] += w_lift_vel * rewards_dict["lifting_vel"]
        if w_close > 0.0:
            self.episode_sums["reward_close"] += w_close * rewards_dict["close"]
        if w_goal > 0.0:
            self.episode_sums["reward_goal"] += w_goal * rewards_dict["goal"]
        if w_goal_fine > 0.0:
            self.episode_sums["reward_goal_fine"] += w_goal_fine * rewards_dict["goal_fine"]
        if w_action != 0.0:
            self.episode_sums["reward_action_rate"] += w_action * rewards_dict["action_rate"]
        if w_joint != 0.0:
            self.episode_sums["reward_joint_vel"] += w_joint * rewards_dict["joint_vel"]
        if w_torques != 0.0:
            self.episode_sums["reward_torques"] += w_torques * rewards_dict["torques"]

        # Store for curriculum logging
        # 1. Is end-effector close to object?
        is_reached = mdp_rewards.object_is_reached(self, threshold=self.cfg.reward_settings["object_is_reached_threshold"], object_cfg=SceneEntityCfg("object"))
        
        
        # 2. Is object robustly lifted? (Height > 0.05m AND Dist < 0.15m to EE)
        # Using new stable check to filter out "throwing", "balancing", and penetration
        is_lifted = mdp_rewards.check_stable_lifting_success(
            self, 
            min_height=self.cfg.success_criteria["min_height"],
            min_grasp_quality=self.cfg.success_criteria["min_grasp_quality"],
            velocity_threshold=self.cfg.success_criteria["velocity_threshold"],
            min_distance_to_ee=self.cfg.success_criteria["min_distance_to_ee"],
            max_distance_to_ee=self.cfg.success_criteria["max_distance_to_ee"],
            object_cfg=SceneEntityCfg("object"),
            ee_frame_cfg=SceneEntityCfg("ee_frame"),
            joint_name="r_joint"
        )
        
        # Update Anytime Consecutive Success Tracker
        # 1. Update counter: increment if lifted, else reset to 0
        self.consecutive_success_counter = (self.consecutive_success_counter + 1) * is_lifted.long()
        
        # 2. Check strict success: held for N steps
        is_success_now = (self.consecutive_success_counter >= self.consecutive_success_threshold)
        
        # 3. Check time validity: ignore initial N steps of the episode
        # We need episode length. self.episode_length_buf tracks current step count
        is_valid_time = (self.episode_length_buf > self.initial_ignore_steps)
        
        # 4. Latch success: marked as successful if valid condition met ANY time
        self.episode_max_success = torch.maximum(self.episode_max_success, (is_success_now & is_valid_time).float())

        # 3. Is object at target position? (direct workflow: use target_poses)
        # For Direct workflow, we use target_pos parameter in object_goal_is_tracked
        # CRITICAL FIX: Add env_origins here too
        # Note: target_pos_w was computed above in this method
        is_tracked = mdp_rewards.object_goal_is_tracked(
            self, 
            threshold=self.cfg.reward_settings.get("goal_tracking_threshold", 0.08), 
            minimal_height=self.cfg.reward_settings["goal_min_height"], 
            target_pos=target_pos_w, 
            object_cfg=SceneEntityCfg("object")
        )

        # Store metrics in extras for curriculum and logging
        if not hasattr(self, "extras"):
            self.extras = {}
        if "episode" not in self.extras:
            self.extras["episode"] = {}

        # Store metrics: reaching, lifting, and goal tracking success
        # OPTIMIZATION: Removed .item() to prevent CPU-GPU sync every step
        self.extras["episode"]["reaching_success"] = torch.mean(is_reached.float())
        self.extras["episode"]["lifting_success"] = torch.mean(is_lifted.float())
        
        # DIRECT WORKFLOW FIX: object_goal_tracking_success now correctly uses target_poses
        self.extras["episode"]["object_goal_tracking_success"] = torch.mean(is_tracked.float())
        self.extras["episode"]["episode_lifting_success"] = self.current_rolling_success
        
        # Calculate and monitor the mean lift height of the object across all envs
        object_bottom_z = mdp_rewards._get_object_bottom_z(self, SceneEntityCfg("object"))
        self.extras["episode"]["mean_lift_height"] = torch.mean(object_bottom_z).item()
        
        # (Legacy per-step logging removed in favor of accumulated sums in _reset_idx)
        # # Store individual reward term averages for TensorBoard (ISSUE FIX #2, #6)
        # # Only log curriculum-weighted rewards when their weight > 0 (active in current curriculum phase)
        # # This shows which reward terms are actually contributing to the agent's learning
        # if w_reach > 0.0:
        #     self.extras["episode"]["reward_reaching"] = torch.mean(w_reach * rewards_dict["reaching"]).item()
        # if w_lift > 0.0:
        #     self.extras["episode"]["reward_lifting"] = torch.mean(w_lift * rewards_dict["lifting"]).item()
        # if w_lift_vel > 0.0:
        #     self.extras["episode"]["reward_lifting_vel"] = torch.mean(w_lift_vel * rewards_dict["lifting_vel"]).item()
        # if w_close > 0.0:
        #     self.extras["episode"]["reward_close"] = torch.mean(w_close * rewards_dict["close"]).item()
        # if w_goal > 0.0:
        #     self.extras["episode"]["reward_goal"] = torch.mean(w_goal * rewards_dict["goal"]).item()
        # if w_goal_fine > 0.0:
        #     self.extras["episode"]["reward_goal_fine"] = torch.mean(w_goal_fine * rewards_dict["goal_fine"]).item()
        # if w_action != 0.0:  # Action penalty (typically negative)
        #     self.extras["episode"]["reward_action_rate"] = torch.mean(w_action * rewards_dict["action_rate"]).item()
        # if w_joint != 0.0:  # Joint velocity penalty (typically negative)
        #     self.extras["episode"]["reward_joint_vel"] = torch.mean(w_joint * rewards_dict["joint_vel"]).item()

        # NOTE: Curriculum updates moved to _reset_idx() for episode-level success metrics
        # (per-step success rates like 0.1 are not meaningful for curriculum thresholds of 0.6)
        
        # [Global Safety Net] Ensure no NaN/Inf rewards are returned to the RL algorithm
        total_reward = torch.nan_to_num(total_reward, nan=0.0, posinf=0.0, neginf=0.0)
        return total_reward

    def step(self, actions: torch.Tensor) -> tuple[dict, torch.Tensor, bool, bool, dict]:
        """Step the environment (ISSUE FIX #4: Record episode rewards when resetting)."""
        obs, reward, terminated, truncated, extras = super().step(actions)
        
        # Initialize logging structure
        if "log" not in extras:
            extras["log"] = {}
        if "episode" not in extras["log"]:
            extras["log"]["episode"] = {}
        
        # Log action numerical values to verify policy clamping
        extras["log"]["episode"]["action_mean"] = actions.mean().item()
        extras["log"]["episode"]["action_max"] = actions.abs().max().item()
        
        # CRITICAL FIX: Merge metrics computed in _get_rewards() to extras["log"]["episode"]
        # This ensures reaching_success, lifting_success, etc. are properly logged to TensorBoard
        if hasattr(self, "extras") and "episode" in self.extras:
            for key, value in self.extras["episode"].items():
                # Don't override values that come from _reset_idx (like episode_lifting_success)
                # but do merge all per-step metrics from _get_rewards()
                if key not in extras["log"]["episode"] or key in [
                    "reaching_success", "lifting_success", "object_goal_tracking_success",
                    "reward_reaching", "reward_lifting", "reward_lifting_vel", 
                    "reward_close", "reward_goal", "reward_goal_fine",
                    "reward_action_rate", "reward_joint_vel", "mean_lift_height"
                ]:
                    extras["log"]["episode"][key] = value
        
        # Write pending reward logs from _reset_idx to the info dict
        # This ensures they appear in TensorBoard and takes precedence over per-step metrics
        if self._pending_reward_logs:
            for name, value in self._pending_reward_logs.items():
                # Write to rsl_rl accepted episode path
                extras["log"]["episode"][name] = value
            
            # Clear pending logs after writing
            self._pending_reward_logs.clear()
        
        # For compatibility with Curriculum or SKRL, keep a copy in "episode"
        if "log" in extras and "episode" in extras["log"]:
            extras["episode"] = extras["log"]["episode"]

        # Update self.extras (ensure Curriculum reads it next step)
        self.extras = extras
        
        # Update visualization markers at every step
        self._update_debug_vis()
        
            # [DEBUG] Print lifting metrics for Env 0 every 10 steps
            # if self.common_step_counter % 2000 <= 500 and self.common_step_counter % 10 == 0:
            #     env_id = 0
            #     # Re-calculate individual checks for debugging
            #     # Note: current implementation of check_robust_lifting_success returns the final boolean
            #     # We need to peek inside to see which condition fails
                
            #     # 1. Height
            #     object_bottom_z = mdp_rewards._get_object_bottom_z(self, SceneEntityCfg("object"))
            #     h_val = object_bottom_z[env_id].item()
                
            #     # 2. Distance
            #     dist_val = mdp_rewards.object_bbox_ee_distance_real(self, SceneEntityCfg("object"), SceneEntityCfg("ee_frame"))[env_id].item()
                
            #     # 3. Status
            #     # Recalculate lifting status for debug print (since is_lifted is not in scope)
            #     is_lifted_debug = mdp_rewards.check_stable_lifting_success(
            #         self, 
            #         min_height=self.cfg.success_criteria["min_height"],
            #         min_grasp_quality=self.cfg.success_criteria["min_grasp_quality"],
            #         velocity_threshold=self.cfg.success_criteria["velocity_threshold"],
            #         min_distance_to_ee=self.cfg.success_criteria["min_distance_to_ee"],
            #         max_distance_to_ee=self.cfg.success_criteria["max_distance_to_ee"],
            #         object_cfg=SceneEntityCfg("object"),
            #         ee_frame_cfg=SceneEntityCfg("ee_frame"),
            #         joint_name="r_joint"
            #     )
            #     is_lifted_val = is_lifted_debug[env_id].item()
            #     counter_val = self.consecutive_success_counter[env_id].item()
            #     latched_success_val = self.episode_max_success[env_id].item()
                
            #     print(f"[Debug Step {self.common_step_counter}] Env 0: "
            #           f"Height={h_val:.4f}, "
            #           f"Lifted={is_lifted_val}, "
            #           f"Counter={counter_val}/{self.consecutive_success_threshold}, "
            #           f"LatchedSuccess={latched_success_val}, "
            #           f"Rolling={self.current_rolling_success:.4f}")

        
        return obs, reward, terminated, truncated, extras

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute termination conditions."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Out of workspace (ISSUE FIX #2: Objects that exceed bounds should trigger reset)
        # RECTANGULAR TERMINATION FIX: Use custom XY range check
        out_of_bounds = mdp_term.object_out_of_workspace_rect(
            self, rect_range=self.cfg.workspace_range, asset_cfg=SceneEntityCfg("object")
        )

        # Return as (terminated, time_out) where terminated includes out_of_bounds
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        
        # Convert to tensor if needed
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        
        # Collect per-environment metrics BEFORE reset for episode logging (ISSUE FIX #4)
        # Note: Skip metrics collection during initial setup (when all envs are being reset)
        # as the sensors might not be fully initialized
        try:
            is_reached = mdp_rewards.object_is_reached(self, threshold=self.cfg.reward_settings["object_is_reached_threshold"], object_cfg=SceneEntityCfg("object"))
            # Use stable check for episode logging too
            # Use robust check for episode logging too (matches step logic)
            is_lifted = mdp_rewards.check_stable_lifting_success(
                self, 
                min_height=0.06,              # REDUCED: from 0.08 to 0.06 - allow lower lifting
                min_grasp_quality=0.20,       # REDUCED: from 0.25 to 0.20 - easier grasp detection
                velocity_threshold=0.20,      # INCREASED: from 0.15 to 0.20 - more tolerance for movement
                min_distance_to_ee=0.01,      # REDUCED: from 0.02 to 0.01 - allow closer approach
                max_distance_to_ee=0.15,      # INCREASED: from 0.12 to 0.15 - allow more variation
                object_cfg=SceneEntityCfg("object"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                joint_name="r_joint"
            )
            # 3. Tracked check (DIRECT WORKFLOW FIX: Use object_goal_is_tracked with target_poses)
            # This should match the computation in _get_rewards() for consistency
            target_pos_w = self.target_poses + self.scene.env_origins
            
            # Use config values for tracking success check (Issue: Hardcoded 0.03 was too strict)
            # Using 'object_is_reached_threshold' (0.08) or 'goal_std' (0.3) logic? 
            # Ideally, tracking success should be related to 'goal' reward parameters or a specific success criteria.
            # Manager-based commonly uses 0.02 ~ 0.05. 
            # User reported 0.03 is too strict (visually wiggles).
            # Let's use a dedicated success threshold from config if available, or relax it.
            # User added reward_settings["goal_std"]=0.3, goal_min_height=0.20.
            # But for binary success, we need a hard threshold. 
            # Let's look for "tracking_success_threshold" or reuse 'object_is_reached_threshold' (0.08) if appropriate, 
            # or add a new config. 
            # Given the report, 0.08 (8cm) seems reasonable for "reached" but "tracked" usually implies finer.
            # However, if visual is 80-90% and log is 10%, relaxing to 0.08 or 0.10 is likely needed.
            # Let's use 0.10 (10cm) temporarily or add it to config. 
            # BETTER: Use self.cfg.reward_settings["object_is_reached_threshold"] (0.08) as a baseline for "close enough to goal".
            
            # Actually, let's use the same logic as _get_rewards where we might want consistent definition.
            # But _get_rewards uses continuous tanh.
            
            # Let's use 0.10m (10cm) as a robust tracking threshold for now, or use a new config key.
            # The user has 'goal_fine_std' = 0.05.
            # Let's use 0.08 (same as reaching threshold) for consistency.
            
            is_tracked = mdp_rewards.object_goal_is_tracked(
                self, 
                threshold=self.cfg.reward_settings.get("goal_tracking_threshold", 0.08), 
                minimal_height=self.cfg.reward_settings["goal_min_height"], 
                target_pos=target_pos_w, 
                object_cfg=SceneEntityCfg("object")
            )
            
            # Compute episode-level success rates before resetting buffers
            # [OLD] Batch Average (only resetting envs)
            # reach_rate = torch.mean(is_reached[env_ids].float()).item()
            # lift_rate = torch.mean(is_lifted[env_ids].float()).item()
            # track_rate = torch.mean(is_tracked[env_ids].float()).item()
            
            # [NEW] Population-Based Average
            # 1. Update the buffer with the latest results from the resetting environments
            self.env_reaching_success[env_ids] = is_reached[env_ids].float()
            self.env_lifting_success[env_ids] = is_lifted[env_ids].float()
            self.env_goal_success[env_ids] = is_tracked[env_ids].float()
            
            # 2. Compute the mean across the ENTIRE population (all envs)
            # OPTIMIZATION: Removed .item() to prevent CPU-GPU sync
            reach_rate = torch.mean(self.env_reaching_success)
            lift_rate = torch.mean(self.env_lifting_success)
            track_rate = torch.mean(self.env_goal_success)
            
            # Use latched anytime success for this episode
            finished_successes = self.episode_max_success[env_ids]
            
            # Reset success trackers for the resetting environments
            self.consecutive_success_counter[env_ids] = 0
            self.episode_max_success[env_ids] = 0.0
            
            # Store in pending logs for step() method to write to tensorboard (ISSUE FIX #4)
            self._pending_reward_logs = {
                "reaching_success": reach_rate,
                "lifting_success": lift_rate,
                "object_goal_tracking_success": track_rate,
            }

            # ISSUE FIX #6: Log Episode Reward Sums (Total Return) and Mean Step Rewards
            # Create a dictionary of results for the resetting environments
            for key, tensor_sum in self.episode_sums.items():
                # Extract sums for the resetting envs
                finished_sums = tensor_sum[env_ids]
                # Reset the sums for these envs
                tensor_sum[env_ids] = 0.0
                
                if len(finished_sums) > 0:
                    # 1. Episode Sum (Total Return)
                    # We log the mean of the sums for this batch of finished episodes
                    # OPTIMIZATION: Removed .item()
                    reward_sum_mean = torch.mean(finished_sums)
                    self._pending_reward_logs[key] = reward_sum_mean
                    
                    # 2. Mean Step Reward (Sum / Length)
                    # Calculate true average reward per step for each finished episode
                    lengths = self.episode_length_buf[env_ids].float()
                    lengths = torch.clamp(lengths, min=1.0)  # Avoid division by zero
                    
                    mean_rewards = finished_sums / lengths
                    # OPTIMIZATION: Removed .item()
                    reward_step_mean = torch.mean(mean_rewards)
                    
                    self._pending_reward_logs[f"{key}_mean"] = reward_step_mean

            # Update rolling success rate (Curriculum Metric)
            # Use finished_successes (latched anytime success)
            
            # Record the lifting_success rate for this episode
            self.success_history.extend(finished_successes.tolist())
            
            if len(self.success_history) > 0:
                self.current_rolling_success = sum(self.success_history) / len(self.success_history)
            else:
                self.current_rolling_success = 0.0
            
            self._pending_reward_logs["episode_lifting_success"] = self.current_rolling_success
            
            # CRITICAL: Write episode-level metrics to self.extras["episode"] BEFORE curriculum checks
            # This ensures curriculum.success_based_weight() can access reaching_success, lifting_success, etc.
            if "episode" not in self.extras:
                self.extras["episode"] = {}
            
            # Update self.extras["episode"] with all computed metrics
            self.extras["episode"].update({
                "reaching_success": reach_rate,
                "lifting_success": lift_rate,
                "object_goal_tracking_success": track_rate,
                "episode_lifting_success": self.current_rolling_success,
            })
            
            # ===== CURRICULUM UPDATE: Check and update reward weights based on episode success =====
            # Now that metrics are available, update curriculum using episode-level success rates
            for term, settings in self.cfg.curriculum_settings.items():
                mdp_curriculum.success_based_weight(
                    self,
                    term_name=term,
                    target_weight=settings["target"],
                    metric_key=settings["metric"],
                    threshold=settings["threshold"],
                    initial_weight=0.0,
                    dependency_term_name=settings["dependency"],
                    increment=settings.get("increment", None),
                    interval=settings.get("interval", 50),
                    increment_interval=settings.get("increment_interval", None),
                )

            # [DEBUG] Reset Metrics Log
            # Check if we are resetting environment 0 or a batch that includes 0
            # Convert env_ids to list if tensor
            # check_ids = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else env_ids
            # if 0 in check_ids:
            #     print(f"[Debug Reset] Step={self.common_step_counter} Env 0 Finished Episode. "
            #           f"Reach={reach_rate:.2f}, Lift={lift_rate:.2f}, Track={track_rate:.2f}, "
            #           f"RollingSuccess={self.current_rolling_success:.4f}, ")
            #           #f"History={list(self.success_history)}")

        except Exception as e:
            # During initial setup, metrics collection might fail due to uninitialized sensors
            # print(f"[Note] Skipped metrics collection during reset: {type(e).__name__}")
            self._pending_reward_logs = {}
        
        super()._reset_idx(env_ids)
        
        # Additional scene reset (equivalent to reset_scene_to_default event in Manager-Based)
        # Reset all dynamic elements in the scene to ensure clean state
        self._reset_scene_elements(env_ids)

        # Note: MultiUsdFileCfg with random_choice=True will select a random object
        # at spawn time (env_0), then clone_environments copies that to all envs.
        # Each training run will have different objects, but within a run all envs
        # have the same object type (which is acceptable for curriculum learning).

        # Sample new target poses (ISSUE FIX #1: Only for the envs being reset, not all envs)
        self._sample_target_poses(env_ids)
        
        # ========== NEW: Randomize object yaw orientation ==========
        # Vary object orientation to increase training diversity
        self._randomize_object_yaw(env_ids)
        
        # ========== ROBOT RESET LOGIC ==========
        # 1. Initialize buffers with defaults from jetrover.py (default_joint_pos)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        # 2. Determine final arm joint targets (Random vs Default)
        # DEBUG: Print exact state of the flags
        # print(f"[Debug Reset Check] Env {env_ids[:1]}... randomize_arm_init: {self.cfg.randomize_arm_init}, has_targets: {hasattr(self, 'random_arm_targets')}")
        
        if self.cfg.randomize_arm_init and hasattr(self, 'random_arm_targets'):
            # Sample new random poses
            self._randomize_arm_init_positions(env_ids)
            target_arm_pos = self.random_arm_targets[env_ids]
        else:
            # Revert to default configuration (from jetrover.py)
            target_arm_pos = self.robot.data.default_joint_pos[env_ids][:, self._arm_joint_indices]
        
        # 3. Apply targets to BOTH the teleport buffer and the PD controller
        # This prevents the robot from "snapping" from an old target to the new position
        joint_pos[:, self._arm_joint_indices] = target_arm_pos
        self.robot.set_joint_position_target(target_arm_pos, joint_ids=self._arm_joint_indices, env_ids=env_ids)

        # 4. Teleport the robot to the specified state in the physics engine
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        # Reset object
        try:
            object_body = self.scene["object"]
            # default_root_state normally contains world pose relative to default env origins if set that way,
            # but usually it's just the initial state from config.
            # We want to set it relative to EACH environment's origin.
            
            env_origins = self.scene.env_origins[env_ids]
            default_root_state = object_body.data.default_root_state[env_ids].clone() # Clone to avoid modifying defaults
            
            # Calculate Random Position (Match Manager-Based Logic)
            # This ensures objects spawn in clear area away from robot base
            # Robot base is at (0, 0) in local coords, with working area around ±0.2m
            # We place objects at X: [0.15, 0.35] (further away), Y: [-0.3, 0.3]
            # This prevents collision with robot structure
            
            # Generate local offsets in safe area
            pos_x = torch.rand(len(env_ids), device=self.device) * 0.15 + 0.20    # 0.20 ~ 0.35 (safer from robot)
            pos_y = (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.40   # -0.2 ~ 0.2 (wider lateral range)
            
            # Set position relative to environment origin
            # Z is kept from init_state (0.15), XY is randomized
            default_root_state[:, 0] = env_origins[:, 0] + pos_x
            default_root_state[:, 1] = env_origins[:, 1] + pos_y
            # Z position is explicitly kept from default (should be around 0.15 based on init_state in config)
            # This matches Manager-Based behavior where z range is (0.0, 0.0) meaning no randomization
            default_root_state[:, 2] = default_root_state[:, 2]  # Keep original Z
            
            # Apply to simulation
            object_body.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            object_body.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            
            # Randomize object mass if enabled
            if self.cfg.randomize_object_mass:
                # Sample masses
                masses = torch.rand(len(env_ids), device=self.device)
                masses = masses * (self.cfg.object_mass_range[1] - self.cfg.object_mass_range[0]) + self.cfg.object_mass_range[0]
                
                # Get usage views
                physx_view = object_body.root_physx_view
                default_mass = object_body.data.default_mass[env_ids]
                default_inertia = object_body.data.default_inertia[env_ids]

                # Ensure masses shape matches (N, 1)
                if masses.dim() == 1:
                    masses = masses.unsqueeze(-1)
                
                # Set masses (Requires CPU tensors to avoid omni.physx.tensors expected device -1 error)
                physx_view.set_masses(masses.cpu(), env_ids.cpu().to(torch.int32))

                # Recompute inertias (Scale by mass ratio)
                # ratio shape (N, 1)
                ratio = masses / default_mass
                # inertia shape (N, 9)
                new_inertias = default_inertia * ratio
                
                physx_view.set_inertias(new_inertias, env_ids)

            # Randomize object friction based on the active object type
            self._randomize_object_friction(env_ids)

        except Exception:
            pass

    def _setup_friction_materials(self):
        """One-time setup: set gripper fingertip friction to 1.2 for all environments."""
        try:
            import isaaclab.sim as sim_utils_local
            from pxr import UsdShade, UsdPhysics, PhysxSchema, Gf

            stage = sim_utils_local.get_current_stage()
            gripper_links = ["l_out_link", "r_out_link"]

            for env_id in range(self.num_envs):
                for link_name in gripper_links:
                    # Walk the prim and its children to find/create a physics material
                    prim_path = f"/World/envs/env_{env_id}/Robot/{link_name}"
                    prim = stage.GetPrimAtPath(prim_path)
                    if not prim.IsValid():
                        continue

                    # Find existing material binding or create new one
                    mat_path = f"{prim_path}/GripperMaterial"
                    if not stage.GetPrimAtPath(mat_path).IsValid():
                        # Create a new physics material
                        mat_prim = stage.DefinePrim(mat_path, "Material")
                    else:
                        mat_prim = stage.GetPrimAtPath(mat_path)

                    # Apply UsdPhysics.Material schema
                    phys_mat = UsdPhysics.Material.Apply(mat_prim)
                    phys_mat.GetStaticFrictionAttr().Set(1.2)
                    phys_mat.GetDynamicFrictionAttr().Set(1.0)
                    phys_mat.GetRestitutionAttr().Set(0.0)

                    # Bind the material to the collision prim
                    # Isaac Sim looks for physics material on the prim or its children
                    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                    material = UsdShade.Material(mat_prim)
                    binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

            print(f"[INFO] Gripper friction set to 1.2 for {self.num_envs} environments.")
        except Exception as e:
            print(f"[WARN] Could not set gripper friction: {e}")

    def _randomize_object_friction(self, env_ids: torch.Tensor):
        """Randomize object friction per-reset based on the active object's material properties."""
        try:
            import isaaclab.sim as sim_utils_local
            from pxr import UsdShade, UsdPhysics

            stage = sim_utils_local.get_current_stage()

            # Determine active object ID (same for all envs in a run due to MultiAssetSpawner cloning)
            active_id = self.object_id_buf[0].item() if hasattr(self, 'object_id_buf') else -1
            friction_range = OBJECT_FRICTION_MAP.get(int(active_id), (0.5, 0.7, 0.4, 0.6))
            s_min, s_max, d_min, d_max = friction_range

            env_ids_list = env_ids.cpu().tolist()
            for env_id in env_ids_list:
                prim_path = f"/World/envs/env_{env_id}/Object"
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    continue

                mat_path = f"{prim_path}/ObjectMaterial"
                if not stage.GetPrimAtPath(mat_path).IsValid():
                    stage.DefinePrim(mat_path, "Material")
                mat_prim = stage.GetPrimAtPath(mat_path)

                # Sample random friction values within the range
                static_friction = s_min + random.random() * (s_max - s_min)
                dynamic_friction = d_min + random.random() * (d_max - d_min)
                # 確保動摩擦力不超過靜摩擦力，符合物理真實性 (Dynamic Friction <= Static Friction)
                dynamic_friction = min(dynamic_friction, static_friction)

                phys_mat = UsdPhysics.Material.Apply(mat_prim)
                phys_mat.GetStaticFrictionAttr().Set(float(static_friction))
                phys_mat.GetDynamicFrictionAttr().Set(float(dynamic_friction))
                phys_mat.GetRestitutionAttr().Set(0.0)

                binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
                material = UsdShade.Material(mat_prim)
                binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")

        except Exception as e:
            print(f"[WARN] Could not randomize object friction: {e}")

    def _detect_object_id(self):
        """Detect the active object ID by inspecting the USD stage.
        
        In Isaac Lab, MultiAssetSpawnerCfg with random_choice=True selects one object for env_0
        and then clones it to all other environments.
        """
        import isaaclab.sim as sim_utils
        from pxr import Usd, UsdGeom
        
        try:
            stage = sim_utils.get_current_stage()
            # We check env_0 as the source of truth
            prim_path = "/World/envs/env_0/Object"
            prim = stage.GetPrimAtPath(prim_path)
            
            # Default fallback
            detected_id = -1
            
            if prim.IsValid():
                # Method 1: Check for references (The most reliable way to find which USD was picked)
                refs = prim.GetMetadata("references")
                if refs:
                    # refs is a SdfReferenceListOp. We check prependedItems or appendedItems.
                    # It's a property, not a method!
                    items = []
                    if hasattr(refs, 'prependedItems'):
                        items = refs.prependedItems
                    if not items and hasattr(refs, 'appendedItems'):
                        items = refs.appendedItems
                        
                    if items:
                        asset_path = items[0].assetPath
                        # asset_path looks like "/root/ObjectFolder/31/31.usd"
                        import os
                        # Extract the folder name (which is the ID)
                        folder_name = os.path.basename(os.path.dirname(asset_path))
                        if folder_name.isdigit():
                            detected_id = int(folder_name)
                
                # Method 2: Fallback if no ref (e.g. if we are not using USD references)
                if detected_id == -1:
                    # Check for an attribute that Isaac Lab might have set
                    # (Some versions set the 'source_asset' or similar)
                    pass
            
            if detected_id != -1:
                self.object_id_buf[:] = detected_id
                print(f"🆔 [INFO] Detected Object ID from USD Stage: {detected_id}")
            else:
                print(f"⚠️ [WARNING] Could not detect Object ID from USD Stage. It will remain 0.")
        except Exception as e:
            print(f"❌ [ERROR] Object ID detection failed: {e}")
