#!/usr/bin/env python3
"""
Deep debugging script to trace std parameter flow during training.
Shows the exact path where std becomes negative.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any

def analyze_policy_structure(policy_module: nn.Module) -> Dict[str, Any]:
    """Analyze the complete structure of the policy network."""
    print("\n" + "="*80)
    print("POLICY NETWORK STRUCTURE ANALYSIS")
    print("="*80)
    
    info = {
        'parameters': [],
        'buffers': [],
        'modules': [],
        'std_params': []
    }
    
    # 1. Parameters
    print("\n[PARAMETERS]")
    for name, param in policy_module.named_parameters():
        info['parameters'].append({
            'name': name,
            'shape': param.shape,
            'requires_grad': param.requires_grad,
            'min': param.min().item(),
            'max': param.max().item(),
            'mean': param.mean().item()
        })
        print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        print(f"    → min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}")
        
        if 'std' in name.lower():
            info['std_params'].append(name)
    
    # 2. Buffers
    print("\n[BUFFERS]")
    for name, buffer in policy_module.named_buffers():
        info['buffers'].append({
            'name': name,
            'shape': buffer.shape,
            'min': buffer.min().item(),
            'max': buffer.max().item(),
            'mean': buffer.mean().item()
        })
        print(f"  {name}: shape={buffer.shape}")
        print(f"    → min={buffer.min().item():.6f}, max={buffer.max().item():.6f}, mean={buffer.mean().item():.6f}")
        
        if 'std' in name.lower():
            info['std_params'].append(name)
    
    # 3. Modules
    print("\n[SUBMODULES]")
    for name, module in policy_module.named_modules():
        if name == '':
            continue
        module_type = type(module).__name__
        info['modules'].append({'name': name, 'type': module_type})
        print(f"  {name}: {module_type}")
    
    print(f"\n[SUMMARY]")
    print(f"  Total Parameters: {sum(p.numel() for p in policy_module.parameters())}")
    print(f"  Total Buffers: {sum(b.numel() for b in policy_module.buffers())}")
    print(f"  STD-related params found: {info['std_params']}")
    print("="*80 + "\n")
    
    return info

def create_std_monitor_hook(policy_module: nn.Module, std_param_names: List[str]) -> callable:
    """
    Create a hook function that monitors std values during forward/backward passes.
    """
    class StdMonitor:
        def __init__(self):
            self.call_count = 0
            self.history = []
        
        def check_std_values(self, context: str):
            """Check and record std values."""
            self.call_count += 1
            snapshot = {
                'call': self.call_count,
                'context': context,
                'std_values': {}
            }
            
            for name in std_param_names:
                # Navigate the hierarchy to find the parameter
                obj = policy_module
                parts = name.split('.')
                try:
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    final_param = getattr(obj, parts[-1])
                    
                    snapshot['std_values'][name] = {
                        'min': final_param.min().item(),
                        'max': final_param.max().item(),
                        'mean': final_param.mean().item(),
                        'shape': final_param.shape,
                        'has_negative': (final_param < 0).any().item(),
                        'has_zero': (final_param == 0).any().item()
                    }
                except:
                    snapshot['std_values'][name] = {'error': 'Could not access parameter'}
            
            self.history.append(snapshot)
            return snapshot
    
    monitor = StdMonitor()
    return monitor

def print_execution_flow():
    """
    Print the expected execution flow from checkpoint load to error.
    """
    flow = """
╔════════════════════════════════════════════════════════════════════════════════╗
║                         TRAINING EXECUTION FLOW (PPO)                          ║
╚════════════════════════════════════════════════════════════════════════════════╝

[PHASE 1: CHECKPOINT LOADING & INITIALIZATION]
├─ train.py: main() function called
├─ runner.load(checkpoint_path)
│   └─ Loads model_state_dict into policy network
│   └─ std parameter values: POSITIVE (from checkpoint)
│
├─ std_parameter_repair() [OUR FIX]
│   └─ Scans Parameters and Buffers for 'std'
│   └─ Clamps negative/zero values to min=0.01
│   └─ Result: All std values now >= 0.01
│
└─ enforce_std_positive hook registered on runner.alg.update

[PHASE 2: TRAINING LOOP STARTS]
runner.learn(num_learning_iterations=8000, init_at_random_ep_len=True)
│
└─ OnPolicyRunner.learn() [rsl_rl/runners/on_policy_runner.py:149]
    │
    ├─ FOR ITERATION i IN num_learning_iterations:
    │  │
    │  ├─ env.reset() + rollout collection
    │  │  └─ policy.act(obs) with std >= 0.01  ✓ WORKS
    │  │
    │  └─ loss_dict = self.alg.update()  ← EXECUTION ENTERS HERE
    │     │
    │     └─ PPO.update() [rsl_rl/algorithms/ppo.py:249]
    │        │
    │        ├─ Prepare data (observations, actions, returns, etc.)
    │        │
    │        ├─ FOR epoch IN range(num_learning_epochs=4):
    │        │  │
    │        │  ├─ shuffle data into mini batches (num_mini_batches=2)
    │        │  │
    │        │  └─ FOR mini_batch IN mini_batches:
    │        │     │
    │        │     ├─ [STEP 1] policy.act(obs_batch) ← CALLED HERE
    │        │     │  │
    │        │     │  └─ ActorCritic.act() [rsl_rl/modules/actor_critic.py:119]
    │        │     │     │
    │        │     │     ├─ forward pass: mu, std = network(obs)
    │        │     │     │  ⚠️  POTENTIAL ISSUE: std values may have changed
    │        │     │     │     since last enforce_std_positive() call!
    │        │     │     │
    │        │     │     └─ distribution.sample()
    │        │     │        └─ torch.normal(mu, std)
    │        │     │           └─ ❌ ERROR IF ANY std < 0!
    │        │     │
    │        │     ├─ [STEP 2] Compute loss (policy_loss, value_loss, entropy)
    │        │     │
    │        │     ├─ [STEP 3] loss.backward()
    │        │     │  └─ Compute gradients for all parameters
    │        │     │  └─ std.grad is populated
    │        │     │
    │        │     ├─ [STEP 4] optimizer.step()
    │        │     │  └─ ⚠️  THIS IS WHERE THE PROBLEM HAPPENS!
    │        │     │  │
    │        │     │  │  Typical optimization step:
    │        │     │  │    std_new = std_old - lr * std.grad
    │        │     │  │    
    │        │     │  │  If std.grad is large enough:
    │        │     │  │    std_new could become < 0
    │        │     │  │
    │        │     │  └─ Result: std values now NEGATIVE
    │        │     │
    │        │     └─ [STEP 5] optimizer.zero_grad()
    │        │
    │        │  [BACK TO LOOP TOP]
    │        │  Next mini_batch iteration calls policy.act() again
    │        │  ❌ BUT std IS NOW NEGATIVE FROM optimizer.step()!
    │        │
    │        │  ERROR: RuntimeError: normal expects all elements of std >= 0.0
    │        │
    │        └─ Exit from update() with error
    │
    └─ safe_alg_update() [OUR HOOK]
       └─ enforce_std_positive() is called AFTER update() returns
       └─ But error already occurred INSIDE update()!

╔════════════════════════════════════════════════════════════════════════════════╗
║                            ROOT CAUSE IDENTIFIED                               ║
╚════════════════════════════════════════════════════════════════════════════════╝

The enforce_std_positive() hook is called AFTER the entire update() method completes.
However, the error occurs DURING update(), specifically:

1. optimizer.step() makes std negative
2. Next policy.act() call (same mini_batch loop) gets negative std
3. torch.normal() raises RuntimeError

THE FIX MUST BE:
• Apply constraints DURING the backward/step process
• Not after the entire update() completes
• Either:
  A) Insert hook into optimizer.step() itself
  B) Clamp std AFTER every optimizer.step() call
  C) Use softplus or exp wrapper on std to force non-negativity
  D) Modify loss function to penalize negative std values
    """
    print(flow)

if __name__ == "__main__":
    print_execution_flow()
