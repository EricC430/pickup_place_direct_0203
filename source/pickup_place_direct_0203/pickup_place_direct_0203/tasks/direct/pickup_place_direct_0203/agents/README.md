# Task Agents Configuration Registry

This directory contains the reinforcement learning agent configurations and custom policy network modules for the JetRover Pickup and Place task. 

To maintain a clean codebase, configurations and policy structures are organized by RL framework.

---

## 1. Directory Structure

```
agents/
├── README.md (This file)
├── __init__.py (Exposes package-level custom networks)
├── rl_games/ (RL-Games YAML Configurations)
│   ├── __init__.py
│   └── ppo_cfg.yaml (Standard PPO config for RL-Games)
├── skrl/ (SKRL YAML Configurations)
│   ├── __init__.py
│   └── ppo_cfg.yaml (Standard PPO config for SKRL)
└── rsl_rl/ (RSL-RL Python Configurations & Custom Networks)
    ├── __init__.py
    ├── distillation_cfg.py (Policy distillation runner configuration)
    ├── ppo_cfg.py (Standard PPO runner configuration)
    ├── ppo_0426_cfg.py (0426 Delta Action & actuator smoothing configuration)
    ├── ppo_0510_cfg.py (0510 Task-Space Delta IK configuration)
    ├── ppo_asym_cfg.py (Base asymmetric vision configuration)
    ├── ppo_asym_0403_cfg.py (0403 CGN-guided asymmetric vision configuration)
    ├── ppo_asym_fresh_cfg.py (Asymmetric vision configuration for training from scratch)
    ├── ppo_asym_fresh_refined_cfg.py (Refined asymmetric vision configuration for scratch training)
    └── vision_asym_actor_critic.py (Custom Actor-Critic handling TensorDict inputs)
```

---

## 2. Configuration & Framework Mapping

### RL-Games
*   **Config File**: `rl_games/ppo_cfg.yaml`
*   **Gym Registry Entry Point Key**: `rl_games_cfg_entry_point`
*   **Usage**: Used for training policies with the RL-Games framework.

### SKRL
*   **Config File**: `skrl/ppo_cfg.yaml`
*   **Gym Registry Entry Point Key**: `skrl_cfg_entry_point`
*   **Usage**: Used for training policies with the SKRL framework.

### RSL-RL
RSL-RL configurations are modular python classes subclassing `RslRlOnPolicyRunnerCfg` or `RslRlDistillationRunnerCfg`.

| Config Module | Registered Class | Usage / Feature Flag |
| :--- | :--- | :--- |
| `rsl_rl.ppo_cfg` | `PPORunnerCfg` | Standard PPO training baseline. |
| `rsl_rl.ppo_0426_cfg` | `PPORunnerCfg` | Optimized for actuator smoothness: gamma=0.98, lower learning rate (1e-4), and action/observation clamping. |
| `rsl_rl.ppo_0510_cfg` | `PPORunnerCfg` | Task-Space Delta IK version configuration: clip observations and actions, customized obs groups. |
| `rsl_rl.distillation_cfg` | `DistillationRunnerCfg` | Distillation config for mapping expert policy observations to student policy. |
| `rsl_rl.ppo_asym_cfg` | `PPORunnerCfg` | Config for asymmetric actor-critic with 1130-dim actor and 73-dim critic. |
| `rsl_rl.ppo_asym_fresh_cfg` | `PPORunnerCfg` | Training asymmetric vision from scratch. |
| `rsl_rl.ppo_asym_fresh_refined_cfg` | `PPORunnerCfg` | Refined asymmetric vision scratch config with tuned clip params (0.1) and entropy coeff. |
| `rsl_rl.ppo_asym_0403_cfg` | `PPORunnerCfg` | Extended asymmetric vision config supporting CGN-guided critic (89-dim critic space). |

---

## 3. Custom Actor-Critic Module

### VisionAsymActorCritic
*   **Location**: `rsl_rl/vision_asym_actor_critic.py`
*   **Class Name**: `VisionAsymActorCritic`
*   **Purpose**: A custom on-policy actor-critic class designed to process multi-modal, high-dimensional observations (TensorDict with stacked RGB-D frames, Point Clouds, and proprioception inputs) for the student policy, while keeping a flat privileged state vector for the asymmetric critic.
*   **Safety Features**: Integrates robust NaN-to-numerical-zero input-output sanitization, pre-encoder safety clamps, and a clamped minimum standard deviation to prevent policy collapse during training.
