# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
PPO Runner Config for 0403 CGN-Guided Asymmetric Vision Environment.
Extends the 0318 config (rsl_rl_ppo_asym_fresh_refined_cfg) with updated
experiment name and enriched critic observation documentation.

Key difference from 0318:
  - Critic MLP input: 89 dims (vs 73 in 0318)
    • 73  = base privileged obs
    • +7  = EE proprioception (pos_in_base + quat_in_base)
    • +1  = EE-to-object distance (scalar)
    • +8  = CGN grasp gap features (pos_gap + rot_gap + score + valid)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 96
    max_iterations = 1200
    save_interval = 250
    experiment_name = "pickup_place_direct_vision_asym_0403"
    encoder_learning_rate = 5.0e-5

    # Asymmetric Actor-Critic Observation Groups
    obs_groups = {
        "policy": ["policy_proprio", "policy_images", "policy_points", "policy_high_res"],
        "critic": ["critic"],
    }

    clip_actions = 100.0

    # Policy & Value Networks
    # ========== NETWORK SIZE FOR 0403 MODULAR VISION ==========
    # Actor: 1130 dims (same as 0318)
    # Critic: 89 dims (enriched — see module docstring)
    # 512 → 256 → 128 handles both 73 and 89 dim inputs well
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPO Algorithm Parameters (identical to 0318)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.1,
        entropy_coef=0.005,
        num_learning_epochs=4,
        num_mini_batches=2,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )
