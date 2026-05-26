# Testing Guide for STD Recovery Fixes

## Pre-Test Checklist

Before testing the fixed train.py, verify:

- [ ] train.py has been replaced with the fixed version
- [ ] train_clean.py exists as a reference
- [ ] You have a valid checkpoint to resume from
- [ ] GPU memory is sufficient (~50GB for 2x RTX A6000)
- [ ] Distributed training is configured correctly

## Test 1: Verify Script Versions

### Command
```bash
cd /workspace/test_isaaclab/pickup_place_direct_0203/scripts/rsl_rl

# Check which scripts exist
ls -lh train*.py

# Verify clean version (should NOT have install_std_protection)
grep -c "install_std_protection" train_clean.py
# Expected output: 0

# Verify recovery version (should have install_std_protection)
grep -c "install_std_protection" train.py
# Expected output: 1

# Verify recovery version has NaN handling
grep -c "isnan" train.py
# Expected output: should be multiple
```

### Expected Output
```
train_clean.py                  # Clean version (no protection)
train.py                        # Fixed version (with protection)
train_with_vision_recovery.py   # Backup/template version
train_old_broken.py             # Old broken version
```

## Test 2: Quick Syntax Check

### Command
```bash
# Check Python syntax
python3 -m py_compile scripts/rsl_rl/train.py
python3 -m py_compile scripts/rsl_rl/train_clean.py

# Should complete without errors
echo "✓ Syntax check passed"
```

## Test 3: Dry Run (No Training)

### Command
```bash
# Check if scripts can be imported
python3 -c "
import sys
sys.path.insert(0, 'scripts/rsl_rl')
try:
    import train as train_module
    print('✓ train.py imports successfully')
    if hasattr(train_module, 'install_std_protection'):
        print('✓ install_std_protection function found')
    else:
        print('✗ install_std_protection function NOT found')
except Exception as e:
    print(f'✗ Import failed: {e}')
"
```

## Test 4: Actual Training Resume Test (FULL TEST)

### Setup
```bash
cd /workspace/test_isaaclab/pickup_place_direct_0203

# Verify checkpoint exists
ls -lh logs/rsl_rl/pickup_place_direct_vision_asym/2026-03-17_15-02-31/model_700.pt
# Should show: -rw-r--r-- ... model_700.pt
```

### Run Command (with protection)
```bash
# Run with the FIXED train.py (with STD protection)
./isaaclab.sh -p -m torch.distributed.run --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task Pickup-Place-Direct-Vision-Asym-v2 \
  --num_envs 16 \
  --headless \
  --video \
  --enable_cameras \
  --video_interval 4000 \
  --video_length 750 \
  --resume \
  --load_run 2026-03-17_15-02-31 \
  --checkpoint model_700.pt \
  --distributed \
  2>&1 | tee test_run_with_protection.log
```

### Expected Success Indicators

#### 1. Initialization Phase (First 30 seconds)
Look for:
```
[INFO] Installing STD Parameter Protection System (with NaN/Inf handling)...
[INFO] Protected parameter: std, shape=torch.Size([6])
[INFO] Protected buffer: critic_obs_normalizer._std, shape=torch.Size([1, 73])
[INFO] Repairing std values in loaded checkpoint...
[INFO]   Fixed std: min 0.003155 → 0.010000
[INFO]   Fixed critic_obs_normalizer._std: min 0.000000 → 0.010000
```

#### 2. Protection Activation
```
[INFO] ═══════════════════════════════════════════════════════════════
[INFO] STD PARAMETER PROTECTION SYSTEM ACTIVE
[INFO] ───────────────────────────────────────────────────────────────
[INFO] Valid std range: [0.01, 1.0]
[INFO] Protected parameters: 2
[INFO] Protection features:
[INFO]   ✓ NaN/Inf detection and recovery
[INFO]   ✓ Parameter-level clamping
[INFO]   ✓ Optimizer step protection
[INFO]   ✓ Pre-check validation in act()
[INFO]   ✓ Emergency recovery mechanism
[INFO] ═══════════════════════════════════════════════════════════════
[INFO] Hooked optimizer.step() with NaN-aware std protection
[INFO] Installed act() wrapper with pre/post validation
```

#### 3. Training Progress (After ~2 minutes)
```
# Training should start and continue without RuntimeError
# You should see episode rewards, losses, etc.
```

#### 4. NO Error Messages Like
```
✗ [WARN] Could not clamp critic_obs_normalizer._std: 
         Inplace update to inference tensor outside InferenceMode...
  
✗ [DEBUG] Step 2: std clamped nan → nan

✗ RuntimeError: normal expects all elements of std >= 0.0
```

### Verification After Run
```bash
# Check for success
grep -E "STD PARAMETER PROTECTION SYSTEM ACTIVE" test_run_with_protection.log
# Should output: 2 matches (one per GPU rank)

# Check for errors (should be empty)
grep -E "Could not clamp cricket_obs_normalizer|Step.*std clamped nan" test_run_with_protection.log
# Should output: nothing

# Check for RuntimeError (should be empty)
grep "RuntimeError" test_run_with_protection.log | grep -v "Expected"
# Should output: nothing

# Check training progress
tail -20 test_run_with_protection.log | grep -E "Episode|Loss|Reward"
# Should show training metrics
```

## Test 5: Compare with Old Version (Optional)

### Run Command (without protection - clean version)
```bash
# Run with the CLEAN train.py (no STD protection)
# This might fail, but useful for comparison
./isaaclab.sh -p -m torch.distributed.run --nproc_per_node=2 \
  scripts/rsl_rl/train_clean.py \
  --task Pickup-Place-Direct-Vision-Asym-v2 \
  --num_envs 16 \
  --headless \
  --video \
  --enable_cameras \
  --video_interval 4000 \
  --video_length 750 \
  --resume \
  --load_run 2026-03-17_15-02-31 \
  --checkpoint model_700.pt \
  --distributed \
  2>&1 | tee test_run_without_protection.log
```

### Expected Result
```
# With clean.py, you should see the old error:
RuntimeError: normal expects all elements of std >= 0.0

# This confirms that the protection in train.py is what fixes it
```

## Test 6: Log Analysis

### Quick Log Check
```bash
# All-in-one verification
echo "=== Checking for protection activation ==="
grep -c "STD PARAMETER PROTECTION SYSTEM ACTIVE" test_run_with_protection.log
# Expected: 2 (one per rank)

echo "=== Checking for protection hooks ==="
grep -c "Hooked optimizer.step()" test_run_with_protection.log
# Expected: 2

echo "=== Checking for errors ==="
grep -c "Could not clamp critic_obs_normalizer" test_run_with_protection.log
# Expected: 0 (none)

echo "=== Checking for NaN events ==="
grep -c "has NaN/Inf - recovering" test_run_with_protection.log
# Expected: 0 (normal case) or any number (means recovery worked)

echo "=== Checking for RuntimeError ==="
grep "RuntimeError" test_run_with_protection.log | grep -v "Expected" | wc -l
# Expected: 0 (none)
```

## Test 7: Detailed Step-by-Step

### Step 1: Monitor Initialization
```bash
# Watch initialization in real-time
tail -f terminal_result.log | head -100
# Should see:
# - App launcher messages
# - STD protection installation
# - Parameter repair logs
# - Hook installation
```

### Step 2: Monitor First Training Steps
```bash
# After initialization, watch first training steps
tail -f terminal_result.log | grep -A 5 -B 5 "Step"
# Should see training steps without errors
# Might see a few [DEBUG] lines in first 5 steps, then silent
```

### Step 3: Monitor Long-term Stability
```bash
# After 5 minutes, check CPU/GPU usage and training progression
while true; do
  echo "=== $(date) ==="
  tail -5 terminal_result.log
  sleep 30
done
```

## Success Criteria

Training is successful if:

- [ ] Script initialization completes without errors
- [ ] STD PARAMETER PROTECTION SYSTEM ACTIVE message appears
- [ ] No "Inplace update to inference tensor" errors
- [ ] No "RuntimeError: normal expects all elements of std >= 0.0"
- [ ] Training continues past step 2
- [ ] Episode/Loss metrics are being logged
- [ ] No NaN/Inf recovery events (optional - may happen legitimately)
- [ ] GPU memory is stable
- [ ] Training runs for > 5 minutes without crashing

## Failure Analysis

### If you see: "Could not clamp critic_obs_normalizer"
**Action**: This should NOT appear in fixed version
- Verify you're using the correct train.py
- Check file modification time: `ls -lh train.py`

### If you see: "std clamped nan → nan"
**Action**: NaN is occurring, but protection is attempting recovery
- Check console output for recovery messages
- Wait a few steps - it might auto-recover
- If hangs, check gradient magnitude

### If you see: "RuntimeError: normal expects all elements of std >= 0.0"
**Action**: Protection failed to catch the error
- This means the protection system didn't work as expected
- Verify all hooks were installed correctly
- Check logs for "[CRITICAL] RuntimeError" messages

### If training hangs
**Action**: Debug the hanging point
```bash
# Collect stack trace
python3 -m torch.utils.profiler

# Or manually attach debugger
gdb python
(gdb) py-list
(gdb) py-bt
```

## Performance Validation

### Check Training Speed
```bash
# Measure iterations per second
grep -oP 'Step \K[0-9]+' terminal_result.log | tail -2 | \
  awk '{print "Iteration: " $1}'

# Calculate throughput
# Should be similar to before the fix (~50-100 iter/min for 16 envs)
```

### Check Device Utilization
```bash
# Monitor GPU usage
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv --loop-ms=1000

# Expected: >90% GPU utilization, stable memory usage
```

## Cleanup After Testing

```bash
# Remove test logs
rm test_run_*.log

# Or keep them for analysis
mkdir -p test_results
mv test_run_*.log test_results/
```

## Documentation

- See **FIX_SUMMARY.md** for technical overview
- See **TRAIN_SCRIPTS_GUIDE.md** for usage guide
- See **COMPARISON_GUIDE.md** for script comparison

## Quick Test (5 minutes)

For a quick validation without full training:

```bash
# Just run 10 iterations to test initialization and first steps
./isaaclab.sh -p -m torch.distributed.run --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task Pickup-Place-Direct-Vision-Asym-v2 \
  --num_envs 4 \
  --headless \
  --max_iterations 10 \
  --device cuda:0 \
  --not-distributed \
  --resume \
  --load_run 2026-03-17_15-02-31 \
  --checkpoint model_700.pt

# This runs on single GPU with minimal envs, completes in ~5 minutes
# Should show all protection messages and complete without errors
```

## Success Example Output

```
[INFO] Installing STD Parameter Protection System (with NaN/Inf handling)...
[INFO] Protected parameter: std, shape=torch.Size([6])
[INFO] Protected buffer: critic_obs_normalizer._std, shape=torch.Size([1, 73])
[INFO] Repairing std values in loaded checkpoint...
[INFO]   Fixed std: min 0.003155 → 0.010000
[INFO]   Fixed critic_obs_normalizer._std: min 0.000000 → 0.010000
[INFO] ═══════════════════════════════════════════════════════════════
[INFO] STD PARAMETER PROTECTION SYSTEM ACTIVE
[INFO] ───────────────────────────────────────────────────────────────
[INFO] Hooked optimizer.step() with NaN-aware std protection
[INFO] Installed act() wrapper with pre/post validation
[INFO] ═══════════════════════════════════════════════════════════════
[DEBUG] Step 1: std clamped, new range [0.010000, 0.129600]
Episode 1: Reward=12.5, Loss=0.045
Episode 2: Reward=13.2, Loss=0.042
✓ Training successful!
```

## Questions & Troubleshooting

**Q: How do I know if the protection is active?**  
A: Look for the "STD PARAMETER PROTECTION SYSTEM ACTIVE" message in logs

**Q: What if NaN still occurs?**  
A: The [CRITICAL] emergency recovery should catch it. Check logs for recovery messages.

**Q: Is there performance overhead?**  
A: Yes, ~2-3% overhead, but it's worth the safety guarantee

**Q: How do I disable the protection (not recommended)?**  
A: Use train_clean.py instead

**Q: Can I add vision weights loading?**  
A: Yes, see lines ~318-328 in train.py for the placeholder

Good luck with testing! 🚀
