# Task Environment Registry (任務環境註冊清單)

本目錄包含 JetRover 機器人 Pickup-Place 任務的各式 Direct RL 環境與配置。為了維持目錄的整潔與模組化，底層所有的輔助工具與模型權重已整合至 `utils/` 子目錄中。

---

## 核心輔助工具目錄 (`utils/`)

| 檔案名稱 | 功能說明 |
| :--- | :--- |
| `utils/cgn_visualizer.py` | 負責在 Isaac Sim 3D 視埠中繪製與視覺化 Contact-GraspNet 產出的抓取點與點雲。 |
| `utils/diagnostic_utils.py` | 提供訓練期間數值監控與診斷的工具（如 `DiagnosticProbe`），用以追蹤物理異常或爆炸。 |
| `utils/grasp_predictor.py` | 整合 FastSAM 與 Contact-GraspNet，提供基於影像與深度圖的即時 3D 抓取姿勢預測。 |
| `utils/performance_monitor.py` | 提供高精度的效能分析與時間追蹤，用以找出環境執行期間（如 YOLO、PointNet、CGN）的瓶頸。 |
| `utils/vision_encoder.py` | 包含 CNN (Simple/ResNet) 以及 PointNet 兩種編碼器，將影像或點雲編碼為特徵向量。 |
| `utils/yolo_detector.py` | 整合 Ultralytics YOLOv8 偵測器，進行目標物體 2D 偵測並將其投影至 3D 空間。 |
| `utils/jetrover.py` | 定義 JetRover 機器人的 USD 載入路徑、關節限位、執行器及預設關節位置之基礎設定檔。 |
| `utils/yolov8m.pt` | YOLO 偵測器所使用的預訓練權重。 |

---

## 任務環境與註冊清單

以下為本目錄中各任務環境（Environments）與設定檔（Configs）的功能及對應註冊 ID：

### 1. 基礎狀態環境 (State-Based Baseline)
*   **環境檔案**：`pickup_place_direct_0203_env.py`
*   **設定檔**：`pickup_place_direct_0203_env_cfg.py`
*   **註冊 ID**：`Template-Pickup-Place-Direct-0203-Direct-v0`
*   **功能與特色**：
    *   以物件與關節的 Ground Truth 狀態作為觀測輸入之 Baseline 任務。
    *   提供基礎的 reach / lift / close 獎勵，做為 RL 學習的起步標準。

### 2. 靜態輸入與課程學習優化版 (Curriculum & Memory Optimization)
*   **環境檔案**：`pickup_place_direct_0421_env.py`
*   **設定檔**：`pickup_place_direct_0421_env_cfg.py`
*   **註冊 ID**：
    *   `Template-Pickup-Place-Direct-0421-Static-v0`
    *   `Template-Pickup-Place-Direct-0421-Static-Distill-v0` (用於政策蒸餾)
*   **功能與特色**：
    *   針對 2048 等多平行環境進行顯卡顯存（VRAM）與物理參數的優化。
    *   導入適應性速度懲罰課程（Adaptive Velocity Penalty），在 Reaching 成功率達標後才開始約束速度，避免探索受限。

### 3. 微分控制與平滑安全版 (Delta Action & Smooth Control)
*   **環境檔案**：`pickup_place_direct_0426_env.py`
*   **設定檔**：`pickup_place_direct_0426_env_cfg.py`
*   **註冊 ID**：`Template-Pickup-Place-Direct-0426-Delta-v0`
*   **功能與特色**：
    *   **Delta 控制模式**：神經網路輸出增量式的關節位移，而非絕對目標。
    *   **低通濾波器 (EMA)**：在 `_apply_action` 中加入指數移動平均濾波，模擬真實伺服機遲滯，消除高頻抖動。
    *   **平滑性懲罰**：引入 Effort 懲罰與 Jerk 懲罰，減少機器人Bang-Bang控制現象，符合 Sim2Real 物理安全。

### 4. 基礎雙相機視覺環境 (Dual-Camera Vision Baseline)
*   **環境檔案**：`pickup_place_direct_0203_vision_env.py`
*   **設定檔**：`pickup_place_direct_0203_vision_env_cfg.py`
*   **註冊 ID**：`Template-Pickup-Place-Direct-0203-Vision-Direct-v0`
*   **功能與特色**：
    *   配置低解析度（80x128）與高解析度（400x640）雙相機。
    *   將連續影像的 CNN 特徵與本體感覺狀態串接，作為 RL 的觀測輸入。

### 5. 非對稱視覺環境 (Asymmetric Vision with YOLO)
*   **環境檔案**：`pickup_place_direct_0203_vision_asym_env.py`
*   **設定檔**：`pickup_place_direct_0203_vision_asym_env_cfg.py`
*   **註冊 ID**：`Isaac-Pickup-Place-Direct-Vision-Asym-v0`
*   **功能與特色**：
    *   **Option A 非對稱觀測**：Policy (Actor) 僅能看到視覺特徵（4幀堆疊 CNN）與 YOLO 偵測特徵，無 ground truth 狀態；Critic 能讀取 privileged 狀態。
    *   YOLO 偵測特徵附帶顯式的置信度訊號，供 policy 判別物件是否在視野中。

### 6. 非對稱點雲環境 (Asymmetric Point Cloud - 0310)
*   **環境檔案**：`pickup_place_vision_asym_0310_env.py`
*   **設定檔**：`pickup_place_vision_asym_0310_env_cfg.py`
*   **註冊 ID**：`Pickup-Place-Direct-Vision-Asym-v1`
*   **功能與特色**：
    *   利用 PointNet 編碼器即時處理低解析度深度相機反投影的點雲，補強 2D CNN 特徵。
    *   觀測堆疊 13 幀的 Strided 歷史特徵（包含 CNN、PointNet），捕捉時間動態。

### 7. 具觸覺與數值估測優化版 (Contact Sensors & JVel - 0313)
*   **環境檔案**：`pickup_place_vision_asym_0313_env.py`
*   **設定檔**：`pickup_place_vision_asym_0313_env_cfg.py`
*   **註冊 ID**：
    *   `Pickup-Place-Direct-Vision-Asym-v2` (自 BC 初始化啟動)
    *   `Pickup-Place-Direct-Vision-Asym-v2_2` (全新 RL 啟動)
*   **功能與特色**：
    *   Critic 引入觸覺力學（Contact Sensor）與摩擦係數回饋。
    *   本體感覺部分，以數值微分形式計算關節速度（JVel），對齊實體感測器回報機制。

### 8. 端到端訓練與防炸防震版 (Trainable Encoders & Safety Clamps - 0318)
*   **環境檔案**：`pickup_place_vision_asym_0318_env.py`
*   **設定檔**：`pickup_place_vision_asym_0318_env_cfg.py`
*   **註冊 ID**：`Pickup-Place-Direct-Vision-Asym-v2_3`
*   **功能與特色**：
    *   **Raw Observation 模式**：輸出未處理的原始影像與點雲數據，允許端到端訓練視覺編碼器。
    *   **預訓練權重加載**：支援自動加載行為複製（Behavior Cloning）之經驗正規化器（Empirical Normalizer）與特徵權重。
    *   **物理防震限位**：為 JPos 與 JVel 設定嚴格物理截斷（±2π rad 與 ±20 rad/s），阻絕 PhysX 重點穿透或崩潰引起的數值爆炸毒化 NN。

### 9. 導引抓取姿勢對齊版 (CGN-Guided Grasp Alignment - 0403)
*   **環境檔案**：`pickup_place_vision_asym_0403_env.py`
*   **設定檔**：`pickup_place_vision_asym_0403_env_cfg.py`
*   **註冊 ID**：`Pickup-Place-Direct-Vision-Asym-v3`
*   **功能與特色**：
    *   **即時 CGN 姿勢推理**：在每集 Reset 時，利用高解析度深度相機運算 Contact-GraspNet，預測多組 3D 抓取位姿並固定於物體局部坐標系中。
    *   **抓取姿勢對齊獎勵**：加入 EE 與最近可行抓取姿勢的距離與旋轉差對齊獎勵，引導手臂進行精準接近。
    *   **擴增 Critic 觀測 (89維)**：增加 EE 相對基座姿態、與最近抓取姿勢的 gap features 以及 CGN 置信度。

### 10. GraspNet Baseline
*   **環境檔案**：`pickup_place_direct_0208_graspnet_env.py`
*   **設定檔**：`pickup_place_direct_0208_graspnet_env_cfg.py`
*   **註冊 ID**：`Template-Pickup-Place-Direct-0208-GraspNet-v0`

### 11. 任務空間增量控制版 (Task-Space Delta IK - 0510)
*   **環境檔案**：`pickup_place_direct_0510_env.py`
*   **設定檔**：`pickup_place_direct_0510_env_cfg.py`
*   **註冊 ID**：`Template-Pickup-Place-Direct-0510-v0`
*   **功能與特色**：
    *   **任務空間增量控制**：動作空間為 7 維（6D 任務空間 Delta 位姿 + 1D 夾爪開合），透過 Differential IK 計算目標關節位置，並限制單步最大位移與旋轉。
    *   **多物體隨機化**：支援多達 15 種物體（塑膠小杯子、塑膠盆、玻璃瓶、不鏽鋼湯匙/刀叉/鐵叉、麥克筆、陶瓷盤、玩具等）的隨機化生成與訓練。
    *   **摩擦係數與質量隨機化**：依據真實物理材質為每種物體設定特有的摩擦係數範圍，並隨機隨集生成質量與表面摩擦力，提高訓練魯棒性與 Sim2Real 遷移能力。

---

## 診斷與偵錯指令

如果您需要互動式地測試 YOLO 偵測與投影計算：
*   **偵錯腳本**：`run_interactive_vision_debug.py` / `interactive_vision_debug.py`
*   **測試指令**：
    ```bash
    python3 run_interactive_vision_debug.py
    ```
