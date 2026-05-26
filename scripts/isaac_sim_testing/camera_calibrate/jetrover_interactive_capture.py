import argparse
from isaaclab.app import AppLauncher

# 1. 配置啟動參數
parser = argparse.ArgumentParser(description="Interactive Point Cloud Capture")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 強制開啟相機渲染
args_cli.enable_cameras = True 

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 導入 Isaac Sim 與 Omniverse 核心庫
import torch
import numpy as np
import open3d as o3d
import carb.input
import omni.appwindow # 修正: 使用 appwindow 獲取鍵盤
import omni.kit.viewport.utility
from isaaclab.sim import SimulationContext, SimulationCfg, GroundPlaneCfg, UsdFileCfg
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils

# --- 輔助類別：處理鍵盤輸入 (修正版) ---
class InputManager:
    def __init__(self):
        self.capture_requested = False
        self.switch_view_requested = False
        
        # 獲取輸入介面
        self.input_interface = carb.input.acquire_input_interface()
        
        # 【修正點】透過 AppWindow 獲取鍵盤 ID，而非直接從 carb 獲取
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        
        # 註冊監聽器
        self.sub_id = self.input_interface.subscribe_to_keyboard_events(
            self.keyboard, self._on_keyboard_event
        )

    def _on_keyboard_event(self, event, *args, **kwargs):
        # 當按下按鍵時 (KEY_PRESS)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # 按下 'C' 鍵 -> 拍照
            if event.input == carb.input.KeyboardInput.C:
                self.capture_requested = True
                print("[Input] Capture Requested (C)")
            # 按下 'V' 鍵 -> 切換視角
            if event.input == carb.input.KeyboardInput.V:
                self.switch_view_requested = True
                print("[Input] Switch View Requested (V)")
        return True

    def reset_flags(self):
        self.capture_requested = False
        self.switch_view_requested = False

    def destroy(self):
        self.input_interface.unsubscribe_to_keyboard_events(self.keyboard, self.sub_id)

def save_training_data(rgb, depth, intrinsic_matrix, segmap=None, filename="data_sample.npz"):
    """
    儲存符合訓練要求的 .npz 檔案
    Keys: 'depth', 'K', 'pc', 'segmap', 'rgb'
    """
    print(f"[INFO] Packaging data into {filename}...")
    
    # 1. 處理 Point Cloud ("pc")
    # 先轉為 Open3D 物件以方便計算座標
    o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth))
    o3d_rgb = o3d.geometry.Image(np.ascontiguousarray(rgb[..., :3])) # 去除 alpha
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, 
        depth_scale=1.0,    # Isaac Sim 輸出即為公尺
        depth_trunc=10.0,   # 截斷距離
        convert_rgb_to_intensity=False
    )
    
    h, w = depth.shape
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    # 取得 Nx3 的點雲座標 (Meters)
    pc_array = np.asarray(pcd.points, dtype=np.float32)

    # 2. 準備字典數據
    data_dict = {
        "depth": depth.astype(np.float32),      # 2D 深度圖 (Meters)
        "K": intrinsic_matrix.astype(np.float32), # 3x3 內參
        "pc": pc_array,                         # Nx3 點雲
        "rgb": rgb[..., :3].astype(np.uint8)    # 額外存 RGB 以便人類除錯
    }

    # 3. 加入 Segmap (如果有)
    if segmap is not None:
        # segmap 通常是 Int 類型的 Class ID
        data_dict["segmap"] = segmap.astype(np.int32)

    # 4. 儲存為 .npz (NumPy Zip)
    # 使用壓縮儲存可以大幅減少檔案大小 (特別是 depth 和 segmap)
    np.savez_compressed(filename, **data_dict)
    
    print(f"[SUCCESS] Saved {filename}")
    print(f"   - Keys: {list(data_dict.keys())}")
    print(f"   - PC Shape: {pc_array.shape}")
    if segmap is not None:
        print(f"   - Segmap Shape: {segmap.shape}")

# --- 點雲生成函數 ---
def save_point_cloud(rgb, depth, intrinsic_matrix, filename_prefix="jetrover_scan"):
    print(f"[INFO] Processing point cloud data...")
    
    # 1. Open3D 處理 (利用它來做反投影，比較方便)
    if rgb.shape[-1] == 4: rgb = rgb[..., :3]
    o3d_rgb = o3d.geometry.Image(np.ascontiguousarray(rgb))
    o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth))
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
    )
    
    h, w, _ = rgb.shape
    fx, fy, cx, cy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

    # ==========================================
    # 【新增功能】輸出為 .npy
    # ==========================================
    # 提取座標點 [N, 3] (float64)
    xyz = np.asarray(pcd.points)
    
    # 提取顏色 [N, 3] (float64, 範圍 0-1)
    colors = np.asarray(pcd.colors)
    
    # 合併為 [N, 6] 的矩陣 -> (x, y, z, r, g, b)
    # 這樣訓練時讀取最方便
    point_cloud_np = np.hstack((xyz, colors))
    
    # 存檔 .npy
    npy_filename = f"{filename_prefix}.npy"
    np.save(npy_filename, point_cloud_np)
    print(f"[SUCCESS] Saved NPY file: {npy_filename} | Shape: {point_cloud_np.shape}")

    # (選用) 同時存一份 .ply 方便您用 MeshLab 看
    ply_filename = f"{filename_prefix}.ply"
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"[SUCCESS] Saved PLY file: {ply_filename}")

def main():
    # 3. 場景設定
    # 【修正點】將 device 設為 "cpu"，解決 GPU 記憶體不足崩潰的問題
    # 對於手動擺姿勢拍照來說，CPU 物理完全足夠
    sim_cfg = SimulationCfg(dt=0.01, device="cpu")
    sim = SimulationContext(sim_cfg)

    # 加入地板
    cfg_ground = GroundPlaneCfg()
    cfg_ground.func("/World/Ground", cfg_ground)

    # 加入 JetRover (請修改您的 USD 路徑)
    robot_usd_path = "/workspace/test_isaaclab/Jetrover/jetrover_isaac_sim/jetrover_real_servo_heavier_gripper.usd"  # <--- 請確認這裡路徑正確
    cfg_robot = UsdFileCfg(usd_path=robot_usd_path)
    cfg_robot.func("/World/JetRover", cfg_robot)

    # ==========================================
    # (C) 加入您的自定義 USD 物體 (貼在這裡)
    # ==========================================
    my_object_path = "/root/ObjectFolder/31/31.usd" # 修改成您的檔案
    cfg_my_obj = UsdFileCfg(usd_path=my_object_path, scale=(1.0, 1.0, 1.0))
    
    # 將物體放置在 X=1.0, Z=0.1 (稍微懸空以免卡在地板)
    cfg_my_obj.func(
        "/World/TargetObj", 
        cfg_my_obj, 
        translation=(1.0, 0.0, 0.1), 
        orientation=(1.0, 0.0, 0.0, 0.0)
    )
    # ==========================================

    # 4. 相機設定
    # 請確認您的 JetRover USD 中有這個路徑，或者修改為正確的 Link 名稱
    # 注意：一定要在既有 Link 下面加一層新的名稱 (例如 /InternalCamera)
    camera_parent = "/World/JetRover/depth_cam_link" # <--- 請修改這行為您的相機 Link 路徑
    camera_prim_path = f"{camera_parent}/InternalCamera"
    
    print(f"[INFO] Spawning camera at: {camera_prim_path}")
    
    camera_cfg = CameraCfg(
        prim_path=camera_prim_path,
        update_period=0.0,
        height=480, width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.4, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.013), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
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
    print("4. 按下 'C': 拍照並儲存點雲。")
    print("-" * 50)

    is_robot_view = False
    default_camera_path = "/OmniverseKit_Persp"

    while simulation_app.is_running():
        # 執行模擬步進
        sim.step()
        
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
            
            # rgb = camera.data.output["rgb"][0].cpu().numpy()
            # depth = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
            # intrinsic = camera.data.intrinsic_matrices[0].cpu().numpy()
            
            # filename = f"jetrover_scan_{np.random.randint(1000,9999)}"
            # save_point_cloud(rgb, depth, intrinsic, filename_prefix=filename)

            rgb = camera.data.output["rgb"][0].cpu().numpy()
            depth = camera.data.output["distance_to_image_plane"][0].cpu().numpy()
            
            # 獲取 Segmap (這通常是整數 ID)
            # 注意：如果報錯，可能是場景中沒有定義 semantic tags，這裡做個防呆
            segmap = None
            if "semantic_segmentation" in camera.data.output:
                segmap = camera.data.output["semantic_segmentation"][0].cpu().numpy()
            
            intrinsic = camera.data.intrinsic_matrices[0].cpu().numpy()
            
            # 儲存
            filename = f"training_data_{np.random.randint(1000,9999)}.npz"
            save_training_data(rgb, depth, intrinsic, segmap, filename)
            
            input_manager.capture_requested = False

    input_manager.destroy()
    simulation_app.close()

if __name__ == "__main__":
    main()