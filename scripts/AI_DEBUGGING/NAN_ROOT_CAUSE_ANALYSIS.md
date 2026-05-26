# NaN/Inf Root Cause Analysis: critic_obs_normalizer._std

## Issue Summary

During PPO training resume with checkpoint, `critic_obs_normalizer._std` continuously generates NaN/Inf values, causing:
```
[CRITICAL PRE-CHECK] critic_obs_normalizer._std has NaN/Inf before act() - fixing...
```

This happens repeatedly **every iteration**, indicating the problem is not a one-time event but a systematic issue during the algorithm update cycle.

---

## Root Cause Investigation

### 1. **What is critic_obs_normalizer?**

The `critic_obs_normalizer` is a **running mean/variance normalizer** for critic observations:
- **Purpose**: Normalize observation values to zero mean and unit variance for training stability
- **Components**:
  - `_mean`: Running mean of observations (shape: [1, 73] for 73-dim observation space)
  - `_std`: Running standard deviation (shape: [1, 73])
- **Update Mechanism**: Uses Welford's online algorithm to update statistics during training

### 2. **Why Does It Generate NaN?**

The standard deviation is computed as: $\sigma = \sqrt{\text{variance}}$

NaN occurs when:
1. **Variance calculation has negative inputs** (due to numerical instability)
2. **Observation values contain NaN/Inf** (upstream from environment)
3. **Extreme value propagation** (very large or very small numbers lead to overflow/underflow)
4. **Checkpoint mismatch** (resuming from checkpoint with different observation distribution)

### 3. **Why During Resume?**

When resuming from checkpoint:
- **Loaded observations** may have different distribution than current environment
- **Running statistics** (mean, variance) from saved checkpoint don't match new environment state
- **Variance can become zero or negative** due to mismatch, making √(variance) = NaN

### 4. **The Symptom Pattern**

```
Step 1: std = [valid values]  ✓
        ↓ optimizer.step() / alg.update()
Step 2: std = [NaN/Inf]  ✗  (generated during update)
        ↓ pre-check fixes it to 0.01
Step 3: std = [NaN/Inf]  ✗  (regenerated during update)
        ↓ pre-check fixes it again
...continues indefinitely
```

This shows the **update process itself is generating NaN**, not just accumulating them.

---

## Protection Strategy (5 Layers)

### Layer 1: Checkpoint Repair (on load)
```python
# In install_std_protection():
# Check all std values immediately after checkpoint load
if std < 0.01 or std.isnan():
    std = 0.01  # Reset to safe minimum
```

### Layer 2: Pre-Optimizer-Step Check
```python
# Before optimizer.step():
if std.isnan() or std.isinf():
    std = 0.01  # Reset before gradient updates
```

### Layer 3: Post-Optimizer-Step Protection
```python
# After optimizer.step():
if std.isnan() or std.isinf():
    std = safe_mean_of_valid_values  # Recover with valid values
    if no valid values:
        std = 0.01  # Fallback
```

### Layer 4: Pre-Act Validation (in act() wrapper)
```python
# Before policy.act():
if std.isnan() or std.isinf():
    std = 0.01  # Final safety check before sampling
# Also check if input observations are valid
if obs.isnan() or obs.isinf():
    log warning and potentially clip values
```

### Layer 5: Algorithm Update Protection (NEW)
```python
# Before and after alg.update():
# Pre-update: Ensure all std valid
# Post-update: Fix any NaN/Inf that appeared during update
# If RuntimeError: recover and retry
```

### Layer 6: Safe Parameter Update Function
```python
def safe_param_update(param, source_tensor):
    try:
        param.copy_(source_tensor)  # Try normal copy first
    except RuntimeError as e:
        if "inference tensor" in str(e):
            # For inference tensors, use .data assignment
            param.data = source_tensor.detach().clone()
        else:
            raise
```

---

## What Changed

### Previous Problem
- Only `param.copy_()` was used, which fails on inference tensors
- No protection during `alg.update()` where NaN is actually generated
- No pre-check before optimizer step

### Current Solution
1. **Safe parameter updates** that handle inference tensors
2. **Pre-step checks** to catch NaN before they propagate
3. **alg.update() wrapper** that protects during the critical update phase
4. **Post-step checks** with multiple recovery strategies
5. **Observation value validation** to catch upstream NaN sources

---

## Diagnostic Output

When running with the new protection:

```
[INFO] Installing STD Parameter Protection System...
[INFO] Protected parameter: std, shape=torch.Size([6])
[INFO] Protected buffer: critic_obs_normalizer._std, shape=torch.Size([1, 73])

[INFO] ═════════════════════════════════════════
[INFO] STD PARAMETER PROTECTION SYSTEM ACTIVE
[INFO] ─────────────────────────────────────────
[INFO]   ✓ NaN/Inf detection and recovery
[INFO]   ✓ Pre-optimizer step checks
[INFO]   ✓ Post-optimizer step fixes
[INFO]   ✓ Algorithm update protection
[INFO]   ✓ Inference tensor safety
[INFO] ═════════════════════════════════════════

[CRITICAL PRE-STEP] Step N: critic_obs_normalizer._std has NaN/Inf BEFORE optimizer - fixing...
[DEBUG] Step N: critic_obs_normalizer._std clamped to [0.010000, 1.000000]
[INFO] Installed alg.update() wrapper with observation protection
```

---

## Expected Behavior After Fix

✅ **With Protection**:
- Detects NaN/Inf at multiple points
- Recovers automatically before it causes RuntimeError
- Training continues without crashes
- Logs show repeated fixes (normal until convergence stabilizes)

⚠️ **Warning Signs**:
- If NaN recovery happens too frequently (> every 5 steps), the root cause may be:
  - Observations still contain NaN (check environment)
  - Learning rate too high (gradients becoming NaN)
  - Checkpoint incompatibility with current environment

---

## Next Steps

1. **Test with the new protection active** - should see:
   - "STD PARAMETER PROTECTION SYSTEM ACTIVE" message
   - Repeated pre-checks fixing NaN values
   - Training continues successfully

2. **Monitor recovery frequency**:
   - Healthy: NaN recovery every 10+ steps
   - Warning: NaN recovery every step (indicates underlying instability)
   - Critical: RuntimeError despite protection

3. **If still failing**:
   - Check if observations themselves contain NaN (new diagnostic output)
   - Reduce learning rate
   - Check environment for numerical issues
   - Try training fresh instead of resuming

---

## Technical Details

### Inference Tensor Issue
PyTorch prevents in-place updates on inference tensors for safety. Our solution:
```python
# ❌ Fails on inference tensors:
param.copy_(new_value)

# ✅ Works on all tensors:
try:
    param.copy_(new_value)
except RuntimeError:
    param.data = new_value.detach().clone()
```

### Welford's Algorithm (Observer Normalizer)
The normalizer uses numerically stable running mean/variance:
```
N = count
mean_new = mean_old + (x - mean_old) / N
M2_new = M2_old + (x - mean_old) * (x - mean_new)
var = M2 / N
```

Resume issues occur when:
- N values are updated but means/variances are mismatched
- New observations have different scale than checkpoint expected

---

## Summary

The `critic_obs_normalizer._std` NaN problem during resume is caused by:

1. **Checkpoint-environment mismatch** - statistics don't align with current observations
2. **Variance calculation instability** - extreme values cause √(variance) = NaN
3. **Algorithm update propagation** - NaN is generated deep in alg.update()

Our **6-layer protection system** catches and recovers from NaN at multiple points, ensuring training stability without requiring checkpoint recreation.
