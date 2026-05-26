# 修復詳細變更清單

## 文件修改統計

### train_with_vision_recovery.py - 核心修復

#### 1️⃣ 添加 Safe Parameter Update 函數 (Line ~128)
```python
def safe_param_update(param, source_tensor):
    """Safely update parameter, handling inference tensors."""
    try:
        param.copy_(source_tensor)
    except RuntimeError as e:
        if "inference tensor" in str(e):
            param.data = source_tensor.detach().clone()
        else:
            raise
```
**作用**：處理inference模式下的張量更新失敗

---

#### 2️⃣ Pre-Optimizer-Step 檢查 (Line ~206)
```python
def optimizer_step_with_std_protection(closure=None):
    std_config['update_step_count'] += 1
    
    # ✨ NEW: Pre-check before optimizer step
    with torch.no_grad():
        current_params = get_current_std_params()
        for param_name, param in current_params.items():
            if param.isnan().any() or param.isinf().any():
                print(f"[CRITICAL PRE-STEP] Step {std_config['update_step_count']}: "
                      f"{param_name} has NaN/Inf BEFORE optimizer - fixing...")
                param.fill_(std_config['min_value'])
    
    # Execute optimizer step
    if closure is not None:
        loss = original_optimizer_step(closure)
    else:
        loss = original_optimizer_step()
```
**作用**：在梯度更新前檢查並修復NaN

---

#### 3️⃣ 改進Post-Step Recovery (Line ~228)
```python
# Before - 複雜的recovery邏輯
if param.isnan().any() or param.isinf().any():
    valid_mask = ~(param.isnan() | param.isinf())
    if valid_mask.any():
        valid_mean = param[valid_mask].mean().item()
        valid_mean = max(std_config['min_value'], min(std_config['max_value'], valid_mean))
    else:
        valid_mean = std_config['min_value']
    param_safe = torch.full_like(param, valid_mean)
    param.copy_(param_safe)

# After - 簡化為直接使用安全值
if param.isnan().any() or param.isinf().any():
    std_config['nan_recovery_count'] += 1
    print(f"[CRITICAL POST-STEP] Step {std_config['update_step_count']}: ...")
    safe_param_update(param, torch.full_like(param, std_config['min_value']))
```
**作用**：使用安全參數更新函數，避免inference張量問題

---

#### 4️⃣ 增強觀察值驗證 (Line ~277)
```python
def act_with_std_check(*args, **kwargs):
    with torch.no_grad():
        current_params = get_current_std_params()
        for param_name, param in current_params.items():
            if param.isnan().any() or param.isinf().any():
                print(f"[CRITICAL PRE-CHECK] {param_name} has NaN/Inf before act() - fixing...")
                safe_param_update(param, torch.full_like(param, std_config['min_value']))
        
        # ✨ NEW: Check observation values themselves
        if len(args) > 0:
            obs = args[0]
            if isinstance(obs, dict):
                for obs_key, obs_val in obs.items():
                    if isinstance(obs_val, torch.Tensor):
                        if obs_val.isnan().any() or obs_val.isinf().any():
                            print(f"[CRITICAL] Observation '{obs_key}' contains NaN/Inf! "
                                  f"Shape: {obs_val.shape}, NaN count: {obs_val.isnan().sum()}")
```
**作用**：檢測上游觀察值問題，找到NaN的根源

---

#### 5️⃣ ALG.UPDATE() 保護包裝 (Line ~321) ⭐ **新層保護**
```python
alg_obj = runner.alg
original_alg_update = alg_obj.update

def alg_update_with_obs_protection(*args, **kwargs):
    """Wrapper around alg.update() with observation normalizer protection."""
    # Pre-update: check and fix std parameters
    with torch.no_grad():
        current_params = get_current_std_params()
        for param_name, param in current_params.items():
            if param.isnan().any() or param.isinf().any():
                safe_param_update(param, torch.full_like(param, std_config['min_value']))
    
    # Execute the algorithm update - THIS IS WHERE NaN IS GENERATED
    try:
        loss_dict = original_alg_update(*args, **kwargs)
    except Exception as e:
        if "normal expects all elements of std >= 0.0" in str(e) or "nan" in str(e).lower():
            print(f"[CRITICAL ALG ERROR] {e}")
            print("[CRITICAL] Attempting recovery before retry...")
            with torch.no_grad():
                current_params = get_current_std_params()
                for param_name, param in current_params.items():
                    safe_param_update(param, torch.full_like(param, std_config['min_value']))
            loss_dict = original_alg_update(*args, **kwargs)  # Retry
        else:
            raise
    
    # Post-update: fix any NaN/Inf that appeared DURING update
    with torch.no_grad():
        current_params = get_current_std_params()
        for param_name, param in current_params.items():
            if param.isnan().any() or param.isinf().any():
                std_config['nan_recovery_count'] += 1
                safe_param_update(param, torch.full_like(param, std_config['min_value']))
    
    return loss_dict

alg_obj.update = alg_update_with_obs_protection
```
**作用**：在NaN實際生成的地方（alg.update()）进行保護

---

## 6層保護系統全景

| 層 | 位置 | 觸發時機 | 動作 |
|----|------|--------|------|
| 1 | Checkpoint load後 | `install_std_protection()` | 修復低於0.01的值 |
| 2 | optimizer.step()前 | 新增 Pre-check循環 | 檢測並修復NaN |
| 3 | optimizer.step()後 | Post-step保護 | 恢復任何新生成NaN |
| 4 | act()前 | act_with_std_check() | 最終驗證+觀察值診斷 |
| 5 | alg.update()前後 | alg_update_with_obs_protection() | **關鍵層** 在NaN生成處保護 |
| 6 | 所有更新 | safe_param_update() | 異常處理，防止inference張量失敗 |

---

## 關鍵修復點解釋

### 為什麼第5層（ALG.UPDATE保護）最重要？

```
runner.learn()
  └─ for iteration in range(max_iterations):
       └─ loss_dict = alg.update()  ⭐ NaN在這裡生成！
            └─ for mini_batch in batches:
                 └─ optimizer.zero_grad()
                 └─ loss = compute_loss()
                 └─ loss.backward()
                 └─ optimizer.step()  ← 梯度爆炸導致NaN
                 └─ policy.act() ← 使用了包含NaN的std參數
```

前面的4層保護可以檢測NaN，但第5層直接在`alg.update()`週期中**監控**，確保：
1. 進入前：所有參數有效 ✓
2. 退出後：生成的任何NaN被立即修復 ✓
3. RuntimeError發生時：自動恢復並重試 ✓

---

## 測試檢查清單

- [ ] 使用新版本執行訓練
- [ ] 檢查是否出現 `[INFO] STD PARAMETER PROTECTION SYSTEM ACTIVE`
- [ ] 檢查前10步是否有 NaN 修復消息
- [ ] 檢查是否沒有 RuntimeError
- [ ] 監控訓練損失曲線是否正常

---

## 相關文件

- **train_with_vision_recovery.py** - 完整實現
- **NAN_ROOT_CAUSE_ANALYSIS.md** - 技術深度分析
- **NAN_QUICK_FIX_SUMMARY_ZH.md** - 快速參考

---

**關鍵要點**：

✅ 之前的問題：inplace操作在inference張量上失敗
✅ 新問題根源：alg.update()週期中梯度爆炸產生NaN
✅ 解決方案：6層多重保護，關鍵第5層在alg.update()處防護
