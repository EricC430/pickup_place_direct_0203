# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # 24 # 96       # skrl: rollouts
    max_iterations = 1500  # skrl: timesteps=50000. (50000 / 24 ≈ 2083 iterations)
    save_interval = 100  # 200
    experiment_name = "manager_to_direct_test" # "manager_to_direct_test"#"vision_append" #"manager_to_direct_test"#"vision_append" #"manager_to_direct_test" #"initial_test"# #"pickup_place_direct" # Updated experiment name to match task

    # Eliminate UserWarning about missing 'policy' and 'critic' keys in obs_groups
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    # skrl 指令碼中 clip_actions = False，這裡我們放寬限制來模擬 False
    clip_actions = 100.0     
    clip_observations = 100.0

    # 對應 skrl: models (Policy & Value)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                # skrl: initial_log_std=0.0 -> std=exp(0)=1.0
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
        entropy_coef=0.005,           # skrl: entropy_loss_scale
        num_learning_epochs=5,        # 8            # skrl: learning_epochs
        num_mini_batches=4,           # 8            # skrl: mini_batches
        learning_rate=2.0e-4,#1.0e-5,#         # 3.0e-4       # skrl: learning_rate (注意: 這是 1e-4)
        schedule="adaptive",          # skrl: KLAdaptiveLR
        gamma=0.99,                   # skrl: discount_factor
        lam=0.95,                     # skrl: lambda
        desired_kl=0.01,              # skrl: kl_threshold
        max_grad_norm=1.0,            # skrl: grad_norm_clip
    )