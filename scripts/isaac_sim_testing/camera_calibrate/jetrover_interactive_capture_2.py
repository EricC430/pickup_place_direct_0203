import argparse
import os # 新增: 用於路徑處理
from isaaclab.app import AppLauncher

# 1. 配置啟動參數
parser = argparse.ArgumentParser(description="Interactive Point Cloud Capture (Style B)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 強制開啟相機渲染
args_cli.enable_cameras = True 

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 導入 Isaac Sim 與 Omniverse 核心庫
import torch
import torchvision # 新增: 用於儲存圖片
import numpy as np
import carb.input
import omni.appwindow
import omni.kit.viewport.utility
from isaaclab.sim import SimulationContext, SimulationCfg, GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils

import math

# --- 輔助類別：處理鍵盤輸入 ---
class InputManager:
    def __init__(self):
        self.capture_requested = False
        self.switch_view_requested = False
        
        self.input_interface = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        
        self.sub_id = self.input_interface.subscribe_to_keyboard_events(
            self.keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input == carb.input.KeyboardInput.C:
                self.capture_requested = True
                print("[Input] Capture Requested (C)")
            if event.input == carb.input.KeyboardInput.V:
                self.switch_view_requested = True
                print("[Input] Switch View Requested (V)")
        return True

    def reset_flags(self):
        self.capture_requested = False
        self.switch_view_requested = False

    def destroy(self):
        self.input_interface.unsubscribe_to_keyboard_events(self.keyboard, self.sub_id)

# --- 修改核心：B 方式的儲存邏輯 ---
def save_snapshot_style_b(camera, step_count, base_dir="output_data"):
    """
    使用 B 方式 (Isaac Lab 風格) 儲存影像與深度。
    特點：
    1. 建立 step_xxxxx 資料夾
    2. RGB 與 Depth 進行 Min-Max 正規化 (0-1) 以便視覺化 (解決全黑問題)
    3. 額外儲存 depth_raw.npy 供後續精確計算使用
    """
    # 建立目錄結構
    save_dir = os.path.join(base_dir, f"step_{step_count:08d}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f"[INFO] Saving snapshot to: {save_dir}")

    # 在 Standalone 模式下，env_idx 通常為 0
    env_idx = 0 
    
    # --- 1. 處理 RGB ---
    # 獲取 Tensor 資料 (不先轉 CPU/Numpy，保持 Tensor 操作)
    rgb_data = camera.data.output["rgb"][env_idx].clone()

    # A. 移除 Alpha 通道 (H, W, 4) -> (H, W, 3)
    if rgb_data.shape[-1] == 4:
        rgb_data = rgb_data[..., :3]

    # B. 確保是 Float 格式
    rgb_data = rgb_data.float()

    # C. [關鍵] 自動增益/正規化 (Auto-Exposure)
    # 解決 "全黑" 問題：將數值拉伸到 0.0 ~ 1.0
    rgb_min, rgb_max = rgb_data.min(), rgb_data.max()
    if rgb_max > rgb_min:
        rgb_data = (rgb_data - rgb_min) / (rgb_max - rgb_min)
    
    # D. 調整維度以符合 Torchvision (H, W, C) -> (C, H, W)
    # 注意：B 範例中使用 permute(2, 1, 0) 可能是特殊需求，標準存圖應為 (2, 0, 1)
    if rgb_data.shape[-1] == 3:
        rgb_data = rgb_data.permute(2, 0, 1) 

    # E. 存檔 (RGB PNG)
    rgb_path = os.path.join(save_dir, "rgb.png")
    torchvision.utils.save_image(rgb_data, rgb_path)


    # --- 2. 處理 Depth ---
    depth_key = "distance_to_image_plane" # 優先使用 Planar Depth
    if depth_key not in camera.data.output and "depth" in camera.data.output:
        depth_key = "depth"
        
    if depth_key in camera.data.output:
        depth_data = camera.data.output[depth_key][env_idx].clone()

        # --- [B 方式重點] 額外儲存精準深度 (Raw NPY) ---
        precise_depth = depth_data.clone()
        
        # A. 處理無限大 (Inf) -> 設為 0.0
        precise_depth[torch.isinf(precise_depth)] = 0.0
        
        # B. 轉為 Numpy 並存檔
        precise_depth_np = precise_depth.cpu().numpy() # Float32, Unit: Meters
        
        # C. 確保形狀是 (H, W)
        if precise_depth_np.ndim == 3:
            precise_depth_np = precise_depth_np.squeeze(-1)
        
        npy_path = os.path.join(save_dir, "depth_raw.npy")
        np.save(npy_path, precise_depth_np)
        # ------------------------------------------------

        # --- 處理視覺化深度圖 (PNG) ---
        depth_path = os.path.join(save_dir, "depth.png")
        
        # A. 確保是 Float
        depth_data = depth_data.float()

        # B. 處理無限大與正規化
        valid_mask = torch.isfinite(depth_data)
        if valid_mask.any():
            d_min = depth_data[valid_mask].min()
            d_max = depth_data[valid_mask].max()
            
            # 將 Inf 設為最大值以便顯示 (或可設為 0)
            depth_data[~valid_mask] = d_max
            
            # 正規化到 0.0 ~ 1.0
            if d_max > d_min:
                depth_data = (depth_data - d_min) / (d_max - d_min)
            else:
                depth_data = torch.zeros_like(depth_data)
        else:
            depth_data = torch.zeros_like(depth_data)
        
        # C. 處理維度 (H, W, 1) -> (C, H, W)
        if depth_data.ndim == 3 and depth_data.shape[-1] == 1:
            depth_data = depth_data.permute(2, 0, 1) # (1, H, W)
        elif depth_data.ndim == 2:
            depth_data = depth_data.unsqueeze(0)     # (1, H, W)

        torchvision.utils.save_image(depth_data, depth_path)
        print(f"[SUCCESS] Saved images to {save_dir}")

def main():
    # 3. 場景設定
    sim_cfg = SimulationCfg(dt=0.01, device="cpu")
    sim = SimulationContext(sim_cfg)

    # 加入地板
    cfg_ground = GroundPlaneCfg()
    cfg_ground.func("/World/Ground", cfg_ground)

    # 加入 JetRover
    robot_usd_path = "/workspace/test_isaaclab/Jetrover/jetrover_isaac_sim/jetrover_real_servo_heavier_gripper_camera_point.usd"
    cfg_robot = UsdFileCfg(usd_path=robot_usd_path)
    cfg_robot.func("/World/JetRover", cfg_robot)

    # 加入自定義物體
    my_object_path = "/root/ObjectFolder/25/25.usd" # 請依需求修改
    try:
        cfg_my_obj = UsdFileCfg(usd_path=my_object_path, scale=(1.0, 1.0, 1.0))
        cfg_my_obj.func(
            "/World/TargetObj", 
            cfg_my_obj, 
            translation=(0.8, 0.0, 0.1), 
            orientation=(1.0, 0.0, 0.0, 0.0)
        )
    except:
        print("[WARN] Custom object not loaded (Path may be wrong)")

    # 4. 相機設定
    camera_parent = "/World/JetRover/depth_cam_link/camera_mount_marker"
    camera_prim_path = f"{camera_parent}/InternalCamera"
    
    print(f"[INFO] Spawning camera at: {camera_prim_path}")
    
    
    # 計算正確的焦距 (Target 79 degrees)
    target_hfov = 79.0
    aperture = 20.955
    calc_focal_length = aperture / (2 * math.tan(math.radians(target_hfov / 2))) # 結果約為 12.71

    camera_cfg = CameraCfg(
        prim_path=camera_prim_path,
        # 修正 1: 限制更新率為 30 FPS
        update_period=1.0 / 30.0, 
        
        height=400, 
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        
        spawn=sim_utils.PinholeCameraCfg(
            # 修正 2: 使用計算出的正確焦距
            focal_length=calc_focal_length, 
            horizontal_aperture=aperture,
            # 修正 3: 加入物理限制的視距範圍
            clipping_range=(0.2, 2.5) 
        ),
        
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            # 建議: 先改回 Identity 檢查方向，除非你確定 Mount 是歪的
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros"
        ),
    )

    # camera_cfg = CameraCfg(
    #     prim_path=camera_prim_path,
    #     update_period=0.0,
    #     height=400, width=640, # 480 640 96 128
    #     data_types=["rgb", "distance_to_image_plane"], # 深度圖必須
    #     spawn=sim_utils.PinholeCameraCfg(focal_length=10.4, horizontal_aperture=20.955),
    #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )
    camera = Camera(cfg=camera_cfg)

    # 5. 初始化模擬
    try:
        sim.reset()
        print("[INFO] Simulation reset successfully. Physics Engine running on CPU.")
    except Exception as e:
        print(f"[ERROR] Simulation reset failed: {e}")
        return

    input_manager = InputManager()
    
    print("-" * 50)
    print("【操作說明】")
    print("1. 請使用滑鼠操作 Isaac Sim GUI (Physics Inspector) 來移動機器人。")
    print("2. 點擊 Viewport 畫面以確保視窗取得焦點。")
    print("3. 按下 'V': 切換視角 (機器人/預設)。")
    print("4. 按下 'C': 使用 B 方式儲存影像 (RGB PNG, Depth PNG, Raw NPY)。")
    print("-" * 50)

    is_robot_view = False
    default_camera_path = "/OmniverseKit_Persp"
    
    # 用於 B 方式的計數器
    step_counter = 15

    while simulation_app.is_running():
        dt = sim.get_physics_dt()

        sim.step()

        camera.update(dt=dt)
        
        # 處理視角切換
        if input_manager.switch_view_requested:
            viewport = omni.kit.viewport.utility.get_active_viewport()
            if viewport:
                if not is_robot_view:
                    viewport.camera_path = camera_prim_path
                    print("[GUI] Switched to Robot Camera View")
                else:
                    viewport.camera_path = default_camera_path
                    print("[GUI] Switched to Default View")
                is_robot_view = not is_robot_view
            input_manager.switch_view_requested = False

        # 處理拍照請求
        if input_manager.capture_requested:
            # 強制更新相機數據
            camera.update(dt=sim.get_physics_dt())
            
            # 使用新的 B 方式儲存
            step_counter += 1
            save_snapshot_style_b(camera, step_count=step_counter, base_dir="output_snapshots")
            
            input_manager.capture_requested = False

    input_manager.destroy()
    simulation_app.close()

if __name__ == "__main__":
    main()