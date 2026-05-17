import torch
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def success_based_weight(
    env: "DirectRLEnv",
    term_name: str,
    target_weight: float,
    metric_key: str,
    threshold: float,
    initial_weight: float = 0.0,
    dependency_term_name: str = None,
    increment: float = None,
    interval: int = 50,
    increment_interval: int = None,
) -> None:
    """
    Curriculum: progressively adjust reward term weight based on success metric.
    
    Args:
        env: DirectRLEnv instance
        term_name: name of the reward term to adjust (stored in env._curriculum_term_weights)
        target_weight: weight to reach when success threshold is crossed
        metric_key: metric key to monitor (in env.extras["episode"])
        threshold: success rate threshold to trigger promotion
        initial_weight: starting weight (if not already set)
        dependency_term_name: another term that must be active first
        increment: amount to increase weight by (linear warm-up)
        interval: number of steps between checks/updates
        increment_interval: separate interval for weight increment (defaults to interval if None)
    """
    if env.common_step_counter < interval:
        return

    # Check curriculum every `interval` steps
    if env.common_step_counter % interval != 0:
        return

    # Initialize curriculum tracking dict if needed
    if not hasattr(env, "_curriculum_term_weights"):
        env._curriculum_term_weights = {}
    if term_name not in env._curriculum_term_weights:
        env._curriculum_term_weights[term_name] = initial_weight

    # Get current metric value
    current_rate = 0.0
    if hasattr(env, "extras") and "episode" in env.extras:
        if metric_key in env.extras["episode"]:
            val = env.extras["episode"][metric_key]
            if isinstance(val, torch.Tensor):
                current_rate = val.item() if val.numel() == 1 else float(val.mean().item())
            else:
                current_rate = float(val)
            if math.isnan(current_rate):
                current_rate = 0.0

    # Check dependency if specified
    if dependency_term_name is not None:
        dep_weight = env._curriculum_term_weights.get(dependency_term_name, 0.0)
        if dep_weight == 0.0:
            if env.common_step_counter % 1000 == 0:
                print(f"[Curriculum Locked] {term_name} waiting for {dependency_term_name}")
            return
        if current_rate <= threshold:
            if env.common_step_counter % 1000 == 0:
                print(f"[Curriculum Check] {term_name} waiting... Rate: {current_rate:.3f} / {threshold}")
            return
    else:
        if current_rate <= threshold:
            if env.common_step_counter % 1000 == 0:
                print(f"[Curriculum Check] {term_name} waiting... Rate: {current_rate:.3f} / {threshold}")
            return

    # Determine effective increment interval
    eff_incr_interval = increment_interval if increment_interval is not None else interval

    # Update weight if threshold crossed
    if env._curriculum_term_weights[term_name] != target_weight:
        # Check if we align with increment interval
        if env.common_step_counter % eff_incr_interval != 0:
            return

        old_weight = env._curriculum_term_weights[term_name]
        
        if increment is not None:
             # Linear warm-up: increase/decrease by increment towards target_weight
             # Determine direction
             if target_weight > old_weight:
                 # Increasing (e.g., 0.0 -> 5.0)
                 new_weight = min(old_weight + abs(increment), target_weight)
             else:
                 # Decreasing (e.g., 0.0 -> -0.001)
                 new_weight = max(old_weight - abs(increment), target_weight)
             
             # Round to avoid floating point artifacts (e.g. 0.30000000000000004)
             new_weight = round(new_weight, 6)
        else:
             # Instant jump (default behavior)
             new_weight = target_weight
             
        env._curriculum_term_weights[term_name] = new_weight

        # [0415] Smart logging: avoid flood for incremental curricula (e.g. joint_vel ramp-up)
        is_first_unlock = (old_weight == 0.0 and new_weight != 0.0)
        is_reached_target = (new_weight == target_weight)

        if is_first_unlock or is_reached_target or increment is None:
            # Full banner: first activation or instant-jump or target reached
            print(f"!")
            if is_first_unlock:
                print(f"[Curriculum UNLOCKED] {term_name} activated!")
            elif is_reached_target:
                print(f"[Curriculum COMPLETE] {term_name} reached target!")
            else:
                print(f"[Curriculum SUCCESS] {term_name} Promoted!")
            print(f"  - Reason: {metric_key} ({current_rate:.3f}) > {threshold}")
            print(f"  - Weight: {old_weight} -> {new_weight} (target={target_weight})")
            print(f"!")
        else:
            # Brief one-liner for gradual ramp-up steps (suppress noisy banners)
            print(f"[Curriculum +] {term_name}: {old_weight} -> {new_weight} "
                  f"(target={target_weight}, {metric_key}={current_rate:.3f})")
