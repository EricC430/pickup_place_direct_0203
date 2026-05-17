# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Template-Pickup-Place-Direct-0203-Direct-v0",
    entry_point=f"{__name__}.pickup_place_direct_0203_env:PickupPlaceDirect0203Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0203_env_cfg:PickupPlaceDirect0203EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Static input version (0421)
gym.register(
    id="Template-Pickup-Place-Direct-0421-Static-v0",
    entry_point=f"{__name__}.pickup_place_direct_0421_env:PickupPlaceDirect0421Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0421_env_cfg:PickupPlaceDirect0421EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# [0426] Delta Action + Smooth Control version
gym.register(
    id="Template-Pickup-Place-Direct-0426-Delta-v0",
    entry_point=f"{__name__}.pickup_place_direct_0426_env:PickupPlaceDirect0426Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0426_env_cfg:PickupPlaceDirect0426EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_0426_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Static input version (0421) - Distillation
gym.register(
    id="Template-Pickup-Place-Direct-0421-Static-Distill-v0",
    entry_point=f"{__name__}.pickup_place_direct_0421_env:PickupPlaceDirect0421Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0421_env_cfg:PickupPlaceDirect0421EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_distillation_cfg:DistillationRunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Vision version (with RGB + Depth camera)
gym.register(
    id="Template-Pickup-Place-Direct-0203-Vision-Direct-v0",
    entry_point=f"{__name__}.pickup_place_direct_0203_vision_env:PickupPlaceDirect0203VisionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0203_vision_env_cfg:PickupPlaceDirect0203VisionEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Contact GraspNet version
gym.register(
    id="Template-Pickup-Place-Direct-0208-GraspNet-v0",
    entry_point=f"{__name__}.pickup_place_direct_0208_graspnet_env:PickupPlaceDirect0208GraspNetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0208_graspnet_env_cfg:PickupPlaceDirect0208GraspNetEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision version YOLO
gym.register(
    id="Isaac-Pickup-Place-Direct-Vision-Asym-v0",
    entry_point=f"{__name__}.pickup_place_direct_0203_vision_asym_env:PickupPlaceDirect0203VisionAsymEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_direct_0203_vision_asym_env_cfg:PickupPlaceDirect0203VisionAsymEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision version Point Cloud (0310)
gym.register(
    id="Pickup-Place-Direct-Vision-Asym-v1",
    entry_point=f"{__name__}.pickup_place_vision_asym_0310_env:PickupPlaceVisionAsym0310Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_vision_asym_0310_env_cfg:PickupPlaceVisionAsym0310EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision version Point Cloud (0313 - With Contact Sensors & Fixed Physics) (start from BC)
gym.register(
    id="Pickup-Place-Direct-Vision-Asym-v2",
    entry_point=f"{__name__}.pickup_place_vision_asym_0313_env:PickupPlaceVisionAsym0313Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_vision_asym_0313_env_cfg:PickupPlaceVisionAsym0313EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision version Point Cloud (0313 - start from fresh RL)
gym.register(
    id="Pickup-Place-Direct-Vision-Asym-v2_2",
    entry_point=f"{__name__}.pickup_place_vision_asym_0313_env:PickupPlaceVisionAsym0313Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_vision_asym_0313_env_cfg:PickupPlaceVisionAsym0313EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_fresh_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision version Point Cloud (0318 - start from fresh RL - fine tuned pretrained)
gym.register(
    id="Pickup-Place-Direct-Vision-Asym-v2_3",
    entry_point=f"{__name__}.pickup_place_vision_asym_0318_env:PickupPlaceVisionAsym0318Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_vision_asym_0318_env_cfg:PickupPlaceVisionAsym0318EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_fresh_refined_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Asymmetric Vision + CGN-Guided Grasp (0403 - enriched critic with grasp alignment reward)
gym.register(
    id="Pickup-Place-Direct-Vision-Asym-v3",
    entry_point=f"{__name__}.pickup_place_vision_asym_0403_env:PickupPlaceVisionAsym0403Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickup_place_vision_asym_0403_env_cfg:PickupPlaceVisionAsym0403EnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_asym_0403_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)