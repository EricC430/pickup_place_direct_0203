# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24       # skrl: rollouts
    max_iterations = 2000 #4166#2083  #8332 #2000        # skrl: timesteps=50000. (50000 / 24 ≈ 2083 iterations)
    save_interval = 50
    experiment_name = "pickup_place_direct_0510" # Task-Space Delta IK version

    # Eliminate UserWarning about missing 'policy' and 'critic' keys in obs_groups
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # skrl 指令碼中 clip_actions = False，這裡我們不採用，改回1
    clip_actions = 100.0
    #限制「輸入給神經網路的觀測值（Observations）
    clip_observations = 100.0

    # 對應 skrl: models (Policy & Value)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,      # skrl: state_preprocessor: RunningStandardScaler
        critic_obs_normalization=True,     # 同上
        actor_hidden_dims=[256, 128, 64],  # skrl: layers
        critic_hidden_dims=[256, 128, 64], # skrl: layers
        activation="elu",                  # skrl: elu
    )

    # 對應 skrl: agent (PPO)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,#2.0,          # skrl: value_loss_scale
        use_clipped_value_loss=True,  # skrl: clip_predicted_values
        clip_param=0.2,               # skrl: ratio_clip
        entropy_coef=0.005,           # 從 0.01 降到 0.005
        num_learning_epochs=8,        # skrl: learning_epochs
        num_mini_batches=4,           # skrl: mini_batches
        learning_rate=1.0e-4,         # skrl: learning_rate (注意: 這是 1e-4)
        schedule="adaptive",          # skrl: KLAdaptiveLR
        gamma=0.99,                   # skrl: discount_factor
        lam=0.95,                     # skrl: lambda
        desired_kl=0.01,              # skrl: kl_threshold
        max_grad_norm=1.0,            # skrl: grad_norm_clip
    )