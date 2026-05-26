"""
COMPREHENSIVE ANALYSIS: Why std Becomes Negative in PPO Training
=====================================================================

FILE: train.py
PURPOSE: Deep investigation into the std parameter degradation issue
DATE: 2026-03-18
"""

# ============================================================================
# PART 1: EXECUTION FLOW ANALYSIS (From Logs)
# ============================================================================

"""
CHECKPOINT STATE AT LOAD TIME:
────────────────────────────────
From inspect_checkpoint.py output:
  - std: shape=torch.Size([6])
    • mean=0.0471, min=0.0032, max=0.1296 ✓ ALL POSITIVE
  
  - critic_obs_normalizer._std: shape=torch.Size([1, 73])
    • mean=1.2212, min=0.0000, max=18.3255 ✗ HAS ZEROS!

OBSERVATION: The critic_obs_normalizer._std buffer contains ZERO values!


EXECUTION TIMELINE:
──────────────────
[Step 1] runner.load(checkpoint)
  └─ std parameter: [0.0032, ..., 0.1296] ✓
  └─ critic_obs_normalizer._std: has min=0.0

[Step 2] std_parameter_repair() [OUR FIX v1]
  └─ Clamps std parameter: 0.0032 < 0.01? NO (already positive)
  └─ Clamps critic_obs_normalizer._std: 0.0000 → 0.01 ✓
  └─ Result: All std values now >= 0.01

[Step 3] runner.learn() starts
  └─ rollout collection: policy.act() → SUCCESS ✓
  
[Step 4] runner.alg.update() called
  ├─ FOR epoch in range(4):
  │  ├─ FOR mini_batch in mini_batches:
  │  │  ├─ [1] policy.act(obs_batch)
  │  │  │   ├─ forward: mu, std = network(obs)
  │  │  │   │   
  │  │  │   │   CRITICAL: What is the network outputting?
  │  │  │   │   Does the network have ANY non-negativity constraint?
  │  │  │   │   Answer: DEPENDS on network architecture
  │  │  │   │   
  │  │  │   └─ distribution.sample() → torch.normal(mu, std)
  │  │  │
  │  │  ├─ [2] compute_loss()
  │  │  ├─ [3] loss.backward()
  │  │  │   └─ std.grad = d(loss)/d(std) ← gradient w.r.t. std
  │  │  │
  │  │  └─ [4] optimizer.step()
  │  │      ├─ std_new = std_old - lr * std.grad
  │  │      │
  │  │      │ If std.grad > std_old / lr:
  │  │      │   std_new < 0 ❌
  │  │      │
  │  │      └─ Result: std potentially NEGATIVE
  │  │
  │  │ [LOOP CONTINUES - BACK TO STEP 1]
  │  │ Next iteration: policy.act() called
  │  │ BUT: std is now NEGATIVE ❌
  │  │ ERROR: RuntimeError: normal expects all elements of std >= 0.0
  │  │
  │  └─ Exit from loop with error
  │
  └─ Error propagates to safe_alg_update() handler


THE CORE PROBLEM:
─────────────────
In PPO's mini_batch loop:
1. std parameter starts at value > 0
2. optimizer.step() SUBTRACTS gradient: std_new = std_old - lr * std.grad
3. If gradient is large, std_new can become NEGATIVE
4. Next policy.act() call in SAME loop fails
5. No clamping happens between optimizer.step() and next policy.act()


WHY PREVIOUS FIX FAILED:
────────────────────────
The safe_alg_update() wrapper called enforce_std_positive() AFTER 
the entire update() method completed. But the error occurred INSIDE 
update(), before it even got a chance to return.

Timeline of failure:
  optimizer.step()  →  std becomes negative
  next iteration    →  policy.act() tries to use negative std
  ERROR raised      ←  No chance for post-update clamping!


SOLUTION ARCHITECTURE:
──────────────────────
"""

# ============================================================================
# PART 2: MATHEMATICAL ROOT CAUSE
# ============================================================================

"""
GRADIENT DESCENT UPDATE EQUATION:
──────────────────────────────────

Standard SGD/Adam Update:
  θ_new = θ_old - lr * ∇L(θ_old)

For std parameter specifically:
  std_new = std_old - lr * ∂L/∂std
  
Condition for std_new < 0:
  std_old - lr * ∂L/∂std < 0
  ⟹ ∂L/∂std > std_old / lr

EXAMPLE: If std_old = 0.05, lr = 5e-5, optimizer=Adam
  Then: ∂L/∂std > 0.05 / 5e-5 = 1000 causes std_new < 0
  
Adam updates are adaptive but still subject to:
  std_new ≥ std_old - lr * moment ← Can go negative!


THE FUNDAMENTAL ISSUE:
───────────────────────
PyTorch's Normal Distribution expects std > 0:
  torch.normal(loc=mu, scale=std)  ← requires std >= 0

But gradient descent has NO inherent constraint that keeps std > 0:
  ∂(log p(a|s)) / ∂std = ... (complex gradient)
  
The gradient can be ANY value, leading to negative updates.
"""

# ============================================================================
# PART 3: MATHEMATICAL SOLUTION APPROACHES
# ============================================================================

"""
SOLUTION A: Hard Clipping (Numerically Unstable)
─────────────────────────────────────────────────
std_new = max(std_old - lr * grad, min_value)

❌ PROBLEM: Creates hard discontinuities in loss gradient
  → Optimizer gets confused with zero/non-zero gradients
  → Training becomes unstable
  → This is what we tried and it still fails!

WHY IT FAILS:
  When std is at boundary (e.g., 0.01), further negative gradients get clipped
  But the optimizer (especially adaptive ones like Adam) maintain momentum/state
  Next iteration, it tries to push std negative again with accumulated momentum
  Results in oscillation or divergence


SOLUTION B: Softplus / Exponential Wrapper (Recommended)
───────────────────────────────────────────────────────
Instead of optimizing std directly, optimize log_std:

std = softplus(log_std) = log(1 + exp(log_std))
Or: std = exp(log_std)

Now:
  ∂L/∂std is computed via chain rule
  std_new = softplus(log_std_old - lr * ∂L/∂log_std)
  
Since softplus always outputs > 0:
  std_new always > 0 ✓
  No hard clipping needed ✓
  Gradients flow properly ✓

IMPLEMENTATION:
  # In policy network, replace final layer:
  # OLD: self.std_output = nn.Parameter(torch.ones(6) * init_std)
  # 
  # NEW: self.log_std = nn.Parameter(torch.log(torch.tensor(init_std)))
  #      self.std_output = torch.nn.functional.softplus(self.log_std)


SOLUTION C: Gradient Clipping on Loss
──────────────────────────────────────
Add L2 regularization or constraint on std itself:

loss_total = loss_ppo + λ * L2_penalty(std)

Where: L2_penalty(std) = sum((std - target_std)^2)

This penalizes both very small and very large std values
Prevents std from drifting too far from initial value


SOLUTION D: Projected Gradient Descent
───────────────────────────────────────
After optimizer.step(), project std back into feasible region:

std_new = max(std_new, min_value)

WITH proper gradient handling:
  No gradient flows through clamping operation
  Clamping is pure parameter update (not in computation graph)
  ✓ This is what our current implementation tries to do


SOLUTION E: Constrained Optimization (Advanced)
────────────────────────────────────────────────
Use Lagrangian or augmented Lagrangian methods
Handle constraint: std >= min_value as formal constraint
"""

# ============================================================================
# PART 4: RECOMMENDED IMPLEMENTATION PATH
# ============================================================================

"""
IMMEDIATE FIX (for current architecture):
──────────────────────────────────────────
Implement Solution D (Projected Gradient) CORRECTLY:

1. Hook into optimizer.step() DIRECTLY
2. Clamp std IMMEDIATELY AFTER parameter update
3. Before next iteration of mini_batch loop

Key requirement:
  ✓ Clamping must happen BETWEEN optimizer.step() and next policy.act()
  ✗ Cannot happen after entire update() completes


LONG-TERM FIX (architect change):
──────────────────────────────────
Implement Solution B (Softplus wrapper):

1. Refactor policy network to use log_std internally
2. Apply softplus when generating std for distribution
3. Automatically ensures std > 0 via network architecture
4. Much cleaner than post-hoc clamping


IMPLEMENTATION DETAILS (Projected Gradient - CORRECT VERSION):
──────────────────────────────────════════════════════════════

Key insight: We need to modify the PYTORCH OPTIMIZER CLOSURE
They don't expose step(), but we can wrap it properly.

For PyTorch optimizers:
  1. Replace optimizer.step()
  2. Call original step()
  3. IMMEDIATELY clamp parameters in same function
  4. Return result

This ensures atomic operation: step() + clamp = single unit
"""

print(__doc__)
