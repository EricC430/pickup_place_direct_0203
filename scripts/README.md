# Isaac Sim / Isaac Lab Script Registry

This document catalogs all the training, evaluation, verification, and utility scripts in this directory.

---

## 1. RSL-RL PPO Policy Runners (`scripts/rsl_rl/`)

These scripts manage Reinforcement Learning (RL) policy training and evaluation using the **RSL-RL** library.

| Script Name | Purpose | Key Features | Env / Task Compatibility |
| :--- | :--- | :--- | :--- |
| `train_ppo.py` | Main training runner | Re-configures modular optimizers; supports trainable/frozen vision encoders. | **All tasks** (e.g., `0203`, `0421`, `0426` direct/vision/asym versions). |
| `train_ppo_0510.py` | Main training runner for 0510 | Standard training script customized for 0510 Task-Space Delta IK task. | `Template-Pickup-Place-Direct-0510-v0` |
| `train_ppo_bc_startup.py` | RL initialization from demonstrations | Loads BC checkpoint to initialize PPO Actor; performs actor-frozen Critic Warmup. | Tasks **0313 and later** (requires `load_bc_normalization_and_encoders` on the environment). |
| `train_ppo_nan_recovery.py` | Training with error recovery | Active NaN/Inf standard deviation detection/protection; observation-normalizer clamping. | **All tasks** (highly recommended for `0421` & `0426` curriculum runs). |
| `train_ppo_legacy_std_protection.py` | Legacy training baseline | Implements standard basic clamping std protection without NaN recovery. | Main direct state environment task (`0426-v0` and legacy). |
| `play_ppo.py` | Default policy evaluation | Plays a trained checkpoint in the simulator. | Standard **state-based direct tasks** (e.g. `Template-Pickup-Place-Direct-0203-v0`). |
| `play_ppo_0510.py` | Playback and Sim2Real trajectory recorder for 0510 | Teleports env, checks anytime success, and records inference actions/joint targets to CSV/PT files. | `Template-Pickup-Place-Direct-0510-v0` |
| `play_ppo_obs_aligned.py` | Asymmetric/Vision policy evaluation | Forces observation alignment (`use_raw_observations=True`); bypasses optimizer parameters. | **Asymmetric vision tasks** (`0203-Vision-Asym`, `0313`, `0403`, etc.). |
| `cli_args.py` | CLI parser helper | Parses RSL-RL related command-line arguments (resume, checkpoint paths, logger). | Internal utility used by all PPO runners. |

---

## 2. Isaac Sim Verification & Diagnostics (`scripts/isaac_sim_testing/`)

These scripts are used to verify and diagnose specific subsystems, environment setups, and sensors.

### Camera Calibration & Projections (`camera_calibrate/`)
- **`verify_camera_projection.py`**: Projects 3D world target positions onto 2D camera pixel coordinates to verify camera projection math.
  - *Compatibility*: Any environment containing RGBD camera sensors (`0203-Vision`, `0313`, `0403`, etc.).
- **`verify_extrinsics.py`**: Visualizes and validates camera extrinsic transform coordinates relative to the robot base.
  - *Compatibility*: Any environment containing camera sensors.
- **`debug_cam_pose.py`**: Utility to test and inspect the physical placement of simulator cameras.

### Grasping Verification (`grasp_verification/`)
- **`verify_0403_cgn.py`**: Validates Contact GraspNet (CGN) scoring parameter configurations, filters, and predicted pose thresholds.
  - *Compatibility*: Specifically designed for the `0403` asymmetric vision environment (`pickup_place_vision_asym_0403_env.py`).

### Policy Distillation Verification (`policy_verification/`)
- **`verify_teacher.py`**: Bypasses student policy execution and runs teacher actions directly inside the distillation environment to verify teacher performance.
  - *Compatibility*: Distillation environment tasks (e.g., `Template-Pickup-Place-Direct-0421-Static-Distill-v0`).

### Offline Diagnostic Tests (`diagnostic_test/`)
- **`test_diagnostic_math.py`**: Offline unit tests (using mocks) to verify joint error calculations (`JErr`), PointNet point cloud filters, and buffer ordering in `DiagnosticProbe`.
  - *Compatibility*: Offline test (no Isaac Sim required).

### Behavioral Cloning (`behavior_cloning/`)
- **`online_bc/train_online_bc.py` / `train_online_bc_boyu.py`**: Trains online behavioral cloning models (CNN/PointNet encoders + MLP policy) directly from offline demonstrations.
  - *Compatibility*: Asymmetric vision-based tasks (`0203-Vision-Asym`, `0313`, `0403`, etc.).
- **`online_bc/play_bc.py` / `play_bc_boyu.py`**: Evaluates trained behavioral cloning checkpoints in the simulator.
  - *Compatibility*: Asymmetric vision-based tasks.
- **`diagnostic_bc/train_diagnostic_bc.py` / `play_diagnostic_bc.py`**: Trains and tests behavioral cloning models based on state observations instead of raw vision, used for diagnostic baselines.
  - *Compatibility*: State-based tasks.
- **`utils/extract_vision_weights.py`**: Extracts the trained ResNet/PointNet visual encoder weights from a BC checkpoint to be loaded during RL startup.
  - *Compatibility*: BC checkpoints.

### Rigid Body Physics & Configuration (`anti-penetration/` & `jetrover_config/`)
- **`anti-penetration/enhance_convex_decomposition.py`**: Generates and optimizes rigid body collision meshes for complex object manipulation.
- **`jetrover_config/check_init_pose.py`**: Verifies and visualizes the initial configuration and joint state limits of the robot.

---

## 3. General Scripts (`scripts/`)

- **`list_envs.py`**: Prints all registered gymnasium environments in the current Isaac Lab environment.
- **`random_agent.py`**: Runs a random policy in the task environment to check step/reward stability.
- **`zero_agent.py`**: Runs a zero-action policy in the task environment to verify robot default posing and gravity stability.
