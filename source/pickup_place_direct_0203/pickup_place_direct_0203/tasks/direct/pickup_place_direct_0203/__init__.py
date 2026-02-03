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