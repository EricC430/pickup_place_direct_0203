# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24       # skrl: rollouts
    max_iterations = 8000        # skrl: timesteps=50000. (50000 / 24 ≈ 2083 iterations)
    save_interval = 100
    experiment_name = "pickup_place_direct_vision_asym" #"asym_test"# # Asymmetric Vision Experiment

    # Asymmetric Actor-Critic Observation Groups (FLATTENED for rsl_rl RolloutStorage compatibility)
    obs_groups = {
        "policy": ["policy"],
        "critic": ["critic"],
    }

    # skrl 指令碼中 clip_actions = False，這裡我們放寬限制來模擬 False
    clip_actions = 100.0     
    clip_observations = 100.0

    # Policy & Value Networks
    # ========== NETWORK SIZE OPTIMIZATION FOR MODULAR VISION (0318) ==========
    # Architecture Design:
    # 
    # Design Principle:
    # Input observation (high-dimensional vision) → wide hidden layers → gradual compression
    # 
    # Observation Dimensions (Modular Architecture):
    # - Policy (Actor): 1130 dims
    #   • Proprioception(42): JPos(6) + JVel(6) + JErr(6) + Last4Actions(24)
    #   • Vision(1088): 4× frames ResNet(512) + 4× frames PointNet(512) + 1× HighRes context(64)
    # 
    # - Critic: 73 dims (Asymmetric / Privileged Info)
    #   • Robot State: JPos(6) + JVel(6) + Last4Actions(24) = 36
    #   • Privileged: ObjPos(3) + ObjBBox(24) + Target(3) + ContactForces(6) + Friction(1) = 37
    # 
    # Network Architecture Rationale:
    # - Layer 1 (input): 1130 → 512 dims
    #   * Sufficient capacity to merge proprioceptive and vision features.
    #   * Compressed embedding for the subsequent layers.
    # - Layer 2: 512 → 256 dims
    # - Layer 3: 256 → 128 dims
    #   * Graduates to output action (6) or value (1).
    #
    # Architecture Evolution:
    # - Previous: YOLO-based (621 dims)
    # - Current: PointNet + ResNet Modular (1130 dims Actor, 73 dims Critic)
    #
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,                
        actor_obs_normalization=True,      # Closed: Using internal LayerNorm in StudentActor for BC
        critic_obs_normalization=True,      # Open for Critic (starts from scratch)
        # Optimized for 1130-dim (policy) and 73-dim (critic) observation spaces
        # With multi-frame modular vision (4 frames ResNet + 4 frames PointNet)
        actor_hidden_dims=[512, 256, 128],  
        critic_hidden_dims=[512, 256, 128], 
        activation="elu",                  
    )

    # PPO Algorithm Parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,               
        entropy_coef=0.005, 
        num_learning_epochs=4,        
        num_mini_batches=2,           
        learning_rate=1.0e-4,          
        schedule="adaptive",          
        gamma=0.99,                   
        lam=0.95,                     
        desired_kl=0.01,               
        max_grad_norm=1.0,            
    )
