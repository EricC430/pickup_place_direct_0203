# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)

@configclass
class DistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 100
    experiment_name = "manager_to_direct_distill"
    
    # Observe both student policy inputs (static) and teacher policy inputs (dynamic)
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        student_obs_normalization=True,
        teacher_obs_normalization=True,  # Old model used normalization, must match!
        student_hidden_dims=[256, 128, 64],
        teacher_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=2.0e-4,
        gradient_length=15,
        max_grad_norm=1.0,
    )
