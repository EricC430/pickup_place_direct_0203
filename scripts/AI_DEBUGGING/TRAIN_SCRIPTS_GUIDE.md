# Train Script Separation and STD Recovery Guide

## Overview

为了解决PPO训练中的`RuntimeError: normal expects all elements of std >= 0.0`错误，我们已经分离出三个版本的train脚本，并修复了关键的NaN/Inf问题。

## Three Versions Available

### 1. **train_clean.py** (干净版本 - 推荐用于新训练)
- **特点**： 完全没有std保护机制
- **用途**： 从头开始的新训练（不涉及checkpoint恢复）
- **优势**： 最小化，无额外开销
- **运行方式**：
  ```bash
  # 新训练（无checkpoint）
  python -m torch.distributed.run --nproc_per_node=2 scripts/rsl_rl/train_clean.py \
    --task Pickup-Place-Direct-Vision-Asym-v2 \
    --num_envs 16 --headless --video \
    --enable_cameras --video_interval 4000
  ```

### 2. **train.py** (带Vision恢复的版本 - 推荐用于checkpoint恢复) ⭐ **当前使用**
- **特点**：
  - ✅ 改进的NaN/Inf检测和恢复
  - ✅ 避免inference tensor的inplace修改
  - ✅ Vision weights加载占位符
  - ✅ Emergency recovery机制
  - ✅ 严格的参数范围检查 (0.01 - 1.0)
  
- **修复的问题**：
  ```
  原错误：[WARN] Could not clamp critic_obs_normalizer._std: 
          Inplace update to inference tensor outside InferenceMode is not allowed.
  
  修复方式：使用 param.copy_() 而不是 param.clamp_() 
           以避免对inference tensors的inplace修改
  ```

- **运行方式**（恢复checkpoint）：
  ```bash
  # 恢复训练（带修复的std保护）
  python -m torch.distributed.run --nproc_per_node=2 scripts/rsl_rl/train.py \
    --task Pickup-Place-Direct-Vision-Asym-v2 \
    --num_envs 16 --headless --video \
    --enable_cameras --video_interval 4000 \
    --resume --load_run 2026-03-17_15-02-31 --checkpoint model_700.pt \
    --distributed
  ```

### 3. **train_with_vision_recovery.py** (独立的Vision恢复脚本)
- **特点**： 与train.py完全相同（train.py是其副本）
- **用途**： 参考版本，便于对比和维护
- **区别**： 将来可以添加专门的vision weights加载逻辑

## Key Fixes Implemented

### 问题1：Inference Tensor Inplace Modification
**原问题**：
```
[WARN] Could not clamp critic_obs_normalizer._std: 
       Inplace update to inference tensor outside InferenceMode is not allowed.
```

**修复方案**：
```python
# ❌ 旧方法（会失败）
param.clamp_(min=0.01)  # inplace operation

# ✅ 新方法（正确）
param_clamped = torch.clamp(param, min=0.01, max=1.0)
param.copy_(param_clamped)  # non-inplace assignment
```

### 问题2：NaN/Inf 值
**原问题**：
```
[DEBUG] Step 2: std clamped nan → nan
```

**修复方案**：
```python
# 检测并恢复NaN/Inf
if param.isnan().any() or param.isinf().any():
    # 恢复为有效值
    valid_mask = ~(param.isnan() | param.isinf())
    if valid_mask.any():
        valid_mean = param[valid_mask].mean().item()
    else:
        valid_mean = 0.01  # 安全最小值
    
    param_safe = torch.full_like(param, valid_mean)
    param.copy_(param_safe)
```

### 问题3：参数范围检查
**改进**：添加了严格的范围检查和动态参数查询
```python
std_config = {
    'min_value': 0.01,    # 下界
    'max_value': 1.0,     # 上界（新增）
    'nan_recovery_count': 0  # 监控恢复次数
}
```

## STD Protection System Features

当使用 **train.py** 恢复checkpoint时，会自动激活以下保护：

1. **Checkpoint Load Protection**
   - 加载checkpoint时立即修复<0.01的std值
   - 检测并恢复NaN/Inf值

2. **Optimizer Step Protection**
   - 每次optimizer.step()后立即检查std
   - 使用动态参数查询确保获取最新值
   - 自动恢复NaN/Inf

3. **Act() Pre-check**
   - policy.act()前预先验证所有std值
   - 检测和恢复异常值

4. **Emergency Recovery**
   - 如果仍然发生RuntimeError，自动触发emergency recovery
   - 将所有std值重置为安全最小值(0.01)
   - 重试act()调用

## File Structure

```
scripts/rsl_rl/
├── train_clean.py              # 干净版本（用于新训练）
├── train.py                    # ⭐ 当前使用（带vision恢复）
├── train_with_vision_recovery.py  # 备份版本
└── train_old_broken.py         # 旧的问题版本（备份）
```

## Usage Recommendations

### 场景1：新训练（从头开始）
```bash
cd /workspace/test_isaaclab/pickup_place_direct_0203
./isaaclab.sh -p -m torch.distributed.run --nproc_per_node=2 \
  scripts/rsl_rl/train_clean.py \
  --task Pickup-Place-Direct-Vision-Asym-v2 \
  --num_envs 16 --headless --video \
  --enable_cameras --video_interval 4000 --distributed
```

### 场景2：恢复checkpoint（带std保护）
```bash
cd /workspace/test_isaaclab/pickup_place_direct_0203
./isaaclab.sh -p -m torch.distributed.run --nproc_per_node=2 \
  scripts/rsl_rl/train.py \
  --task Pickup-Place-Direct-Vision-Asym-v2 \
  --num_envs 16 --headless --video \
  --enable_cameras --video_interval 4000 \
  --resume --load_run 2026-03-17_15-02-31 --checkpoint model_700.pt \
  --distributed
```

### 场景3：调试（不带Vision恢复）
```bash
# 如果train.py仍有问题，可以尝试train_clean.py
./isaaclab.sh -p -m torch.distributed.run --nproc_per_node=2 \
  scripts/rsl_rl/train_clean.py \
  --task Pickup-Place-Direct-Vision-Asym-v2 \
  --num_envs 16 --headless --video \
  --enable_cameras --video_interval 4000 \
  --resume --load_run 2026-03-17_15-02-31 --checkpoint model_700.pt \
  --distributed
```

## Monitoring and Debugging

### 关键日志输出
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
```

### 若出现错误日志
```
[CRITICAL] Step X: parameter_name has NaN/Inf - recovering...
[CRITICAL] RuntimeError during act(): normal expects all elements of std >= 0.0
[CRITICAL] Attempting emergency std recovery...
```

这表示保护系统已激活并在尝试恢复。

## Vision Weights Loading

`train.py` 中包含了vision weights加载的占位符代码（约318-328行）：

```python
# ====================================================================
# SPECIAL: Load vision weights (optional)
# ====================================================================
try:
    vision_weights_path = "logs/vision_weights_standalone.pt"
    if os.path.exists(vision_weights_path):
        print(f"[INFO] Loading vision weights from: {vision_weights_path}")
        # 这里添加您的vision weights加载逻辑
except:
    pass
```

如果需要加载vision weights，可以：
1. 在该位置添加您的vision weights加载代码
2. 或者复制到train_with_vision_recovery.py中作为专用脚本

## Troubleshooting

### 如果仍然遇到std相关错误

1. **检查std值范围**：
   - 添加 `--debug` 标志查看详细日志
   - 查看checkpoint中原始std值

2. **禁用分布式训练**：
   ```bash
   python scripts/rsl_rl/train.py ... --resume ... # 不使用--distributed
   ```

3. **使用train_clean.py**：
   - 如果问题持续，改用train_clean.py不带任何std保护

4. **检查梯度**：
   - std变成NaN通常意味着梯度爆炸或其他数值问题
   - 可能需要调整学习率或梯度裁剪

## Related Files

- `/workspace/test_isaaclab/pickup_place_direct_0203/scripts/rsl_rl/train_old_broken.py` - 之前有问题的版本
- [STD_PROTECTION_REPORT.md](../DEEP_ANALYSIS.md) - 详细的技术分析
- [INVESTIGATION_SUMMARY.md](../INVESTIGATION_SUMMARY.md) - 完整的问题追踪
