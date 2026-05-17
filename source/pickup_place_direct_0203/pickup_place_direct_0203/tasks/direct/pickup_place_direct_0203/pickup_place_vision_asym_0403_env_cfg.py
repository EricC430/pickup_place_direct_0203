# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import math
from isaaclab.utils import configclass
from .pickup_place_vision_asym_0318_env_cfg import PickupPlaceVisionAsym0318EnvCfg


@configclass
class PickupPlaceVisionAsym0403EnvCfg(PickupPlaceVisionAsym0318EnvCfg):
    """
    Configuration for CGN-Guided Asymmetric Vision Environment (0403 Version).
    
    Extends the 0318 config with Contact-GraspNet integration parameters
    for grasp-alignment reward and enriched critic observations.
    
    Critic observation: 89 dims
      Base (73) + ee_pos_in_base(3) + ee_quat_in_base(4) + ee_to_obj_dist(1)
      + grasp_gap_pos(3) + grasp_gap_rot(3) + grasp_score(1) + grasp_valid(1)
    """

    # ===== CONTACT-GRASPNET CONFIGURATION =====
    # Paths — these are CONTAINER paths when running in Isaac Sim
    cgn_ckpt_dir: str = "/workspace/test_isaaclab/contact_graspnet_pytorch/checkpoints/contact_graspnet"
    fastsam_ckpt_path: str = "/workspace/test_isaaclab/FastSAM/weights/FastSAM-x.pt"
    
    # Optional: package roots for sys.path injection (container paths)
    cgn_root: str = "/workspace/test_isaaclab/contact_graspnet_pytorch"
    fastsam_root: str = "/workspace/test_isaaclab/FastSAM"
    
    # Inference device — defaults to cuda:1 (second A6000)
    cgn_device: str = "cuda:1"

    # Grasp selection parameters
    cgn_top_k: int = 5               # Keep top-K grasps per environment
    cgn_score_threshold: float = 0.1  # Minimum CGN confidence score
    cgn_z_range: tuple = (0.1, 2.5)   # Depth clip for point cloud (widened to 2.5m)
    cgn_width_range: tuple = (0.005, 0.055)  # Gripper width filter (m)
    cgn_proximity_filter: float = 0.3  # Max distance from object for valid grasp (m)
    cgn_arg_configs: list = None       # Populated in __post_init__

    # ===== GRASP ALIGNMENT REWARD =====
    rew_scale_grasp_align: float = 0.5      # Reward weight for grasp alignment
    grasp_align_pos_std: float = 0.05       # Sharpness of position alignment (m)
    grasp_align_rot_weight: float = 0.3     # Weight of orientation component vs position

    # ===== DEBUG & VERIFICATION =====
    cgn_debug_vis: bool = False              # [0408] Enabled for FastSAM verification
    cgn_debug_snapshots: bool = False        # [0408] Enabled for RGB overlay verification
    cgn_debug_snapshot_max_episodes: int = 5 
    cgn_debug_dir: str = "logs/cgn_debug"   # Output directory for debug files

    # ===== ENRICHED CRITIC OBSERVATION =====
    # Override parent's 73 with 89
    critic_observation_space: int = 89

    # ===== OBSERVATION SPACE (override property) =====
    @property
    def observation_space(self) -> dict:
        if getattr(self, "use_raw_observations", False):
            return {
                "policy_proprio": 42,
                "policy_images": (4, 3, 80, 128),
                "policy_points": (4, 1024, 3),
                "policy_high_res": 64,
                "critic": self.critic_observation_space,  # 89
            }
        else:
            return 1130

    def __post_init__(self):
        super().__post_init__()
        # Populate cgn_arg_configs with defaults if not set
        if self.cgn_arg_configs is None:
            self.cgn_arg_configs = [
                "TEST.first_thres:0.05",
                "TEST.second_thres:0.05",
                "TEST.filter_thres:0.005",
            ]
