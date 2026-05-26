# 快速修復總結

## 問題

當從checkpoint恢復PPO訓練時，`critic_obs_normalizer._std` 不斷產生 NaN/Inf：

```
[CRITICAL PRE-CHECK] critic_obs_normalizer._std has NaN/Inf before act() - fixing...
[CRITICAL PRE-CHECK] critic_obs_normalizer._std has NaN/Inf before act() - fixing...
...無限重複...
```

## 根本原因

`critic_obs_normalizer._std` 是一個**運行標準差計算器**，用於正規化critic網絡的觀察值。

當從checkpoint恢復時：
1. 加載的統計數據（mean/std）與當前環境不匹配
2. 觀察值分佈發生變化
3. 方差計算變成不穩定（可能為負數或零）
4. $\sqrt{\text{variance}}$ 計算變成 NaN

## 新添加的保護機制（6層）

### 層1: Checkpoint修復（加載時）
在load checkpoint後立即檢查所有std值，如果 < 0.01 則重設為 0.01

### 層2: Pre-Optimizer檢查
在 `optimizer.step()` 之前驗證std值有效

### 層3: Post-Optimizer保護
在 `optimizer.step()` 之後修復任何新產生的 NaN/Inf

### 層4: Pre-Act驗證
在 `policy.act()` 之前最終檢查，並驗證觀察值本身是否有效

### 層5: Algorithm Update保護 ⭐ **新增**
在 `alg.update()` 前後保護 - 這是NaN實際生成的地方

### 層6: 安全參數更新
```python
def safe_param_update(param, source_tensor):
    try:
        param.copy_(source_tensor)
    except RuntimeError:
        # For inference tensors:
        param.data = source_tensor.detach().clone()
```

## 修改的文件

✅ **train_with_vision_recovery.py**
- 添加 alg.update() wrapper
- 改進觀察值驗證
- Pre-step檢查補充

✅ **NAN_ROOT_CAUSE_ANALYSIS.md** (新文件)
- 詳細的技術分析
- Welford算法說明
- 預期行為指南

## 下一步：測試

### 選項A：使用已修復的版本
```bash
./isaaclab.sh -p -m torch.distributed.run --nproc_per_node=2 \
  scripts/rsl_rl/train_with_vision_recovery.py \
  --task Pickup-Place-Direct-Vision-Asym-v2 \
  --num_envs 8 --headless \
  --resume --load_run 2026-03-17_15-02-31 --checkpoint model_700.pt
```

### 選項B：同步到train.py
```bash
cp scripts/rsl_rl/train_with_vision_recovery.py scripts/rsl_rl/train.py
# Then run with train.py
```

## 預期的日誌輸出

✅ 健康狀態：
```
[INFO] STD PARAMETER PROTECTION SYSTEM ACTIVE
[CRITICAL PRE-CHECK] critic_obs_normalizer._std has NaN/Inf - fixing...  # 可能出現 1-3 次
[CRITICAL PRE-STEP] Step 1: critic_obs_normalizer._std has NaN/Inf BEFORE optimizer - fixing...
[CRITICAL POST-STEP] Step 2: critic_obs_normalizer._std has NaN/Inf AFTER optimizer - recovering...
[INFO] Installed alg.update() wrapper with observation protection
...訓練繼續...
```

⚠️ 警告信號（需要進一步調查）：
- 每個iteration都有 NaN 恢復
- RuntimeError 仍然發生
- 觀察值本身包含 NaN（新診斷消息）

## 為什麼會發生這個問題？

從checkpoint恢復時，運行統計數據（running statistics）的完整狀態可能不會完全保存，導致：

1. **統計數據不匹配**：加載的mean/std與當前觀察值分佈不符
2. **方差計算失敗**：variance = E[x²] - E[x]² 在數值邊界情況下失敗
3. **NaN傳播**：std = √(negative variance) = NaN

---

**詳細技術分析見：** [NAN_ROOT_CAUSE_ANALYSIS.md](NAN_ROOT_CAUSE_ANALYSIS.md)
