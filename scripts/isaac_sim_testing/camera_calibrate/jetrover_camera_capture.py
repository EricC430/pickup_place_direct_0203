import argparse
from isaaclab.app import AppLauncher

# 1. 啟動模擬器 (必須最先執行)
parser = argparse.ArgumentParser(description="JetRover Point Cloud Capture")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 導入必要的函式庫
import torch
import numpy as np
import open3d as o3d
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

def create_point_cloud(rgb, depth, intrinsic_matrix):
    """將數據轉換為 Open3D 點雲"""
    # 建立 Open3D 影像物件
    o3d_rgb = o3d.geometry.Image(np.ascontiguousarray(rgb))
    o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth))
    
    # 建立 RGBD (Isaac Sim 單位為公尺，depth_scale=1.0)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
    )
    
    # 建立內參
    h, w, _ = rgb.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
    
    # 反投影生成點雲
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    return pcd

def main():
    # 3. 設定模擬環境
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 4. 載入場景物件
    # (A) 地板
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/Ground", cfg_ground)
    
    # (B) 匯入 JetRover (假設您的 USD 路徑如下，請修改)
    # 這裡我們將機器人生成在原點
    jetrover_usd_path = "/workspace/test_isaaclab/Jetrover/jetrover_isaac_sim/jetrover_real_servo_heavier_gripper.usd" 
    cfg_robot = sim_utils.UsdFileCfg(
        usd_path=jetrover_usd_path,
        scale=(1.0, 1.0, 1.0),
    )
    cfg_robot.func("/World/JetRover", cfg_robot)

    # (C) 匯入被拍攝的物體 (例如一個箱子，放在機器人前方 1 公尺處)
    cfg_object = sim_utils.CuboidCfg(
        size=(0.2, 0.2, 0.2), 
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
    )
    cfg_object.func("/World/TargetObject", cfg_object, translation=(1.0, 0.0, 0.1))

    # 5. 設定相機 (關鍵步驟)
    # 我們將相機 "掛載" 到機器人的相機 Link 上
    # 請在 Isaac Sim GUI 中確認 JetRover 的相機 Link 名稱，通常是 "camera_link" 或 "depth_cam_link"
    camera_parent_link = "/World/JetRover/depth_cam_link"  # 範例路徑，請依實際結構修改
    
    camera_cfg = CameraCfg(
        prim_path=f"{camera_parent_link}/SimCamera", # 將相機生成為該 Link 的子物件
        update_period=0.0,
        height=480, width=640,
        data_types=["rgb", "distance_to_image_plane"], # 必須使用 distance_to_image_plane
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4, horizontal_aperture=20.955
        ),
        # 重要：座標系修正
        # Isaac Sim 相機預設朝向 -Z，而機器人 Link (URDF) 通常是 X 前或 Z 前
        # 這裡使用 ROS convention (0.5, -0.5, 0.5, -0.5) 將 Isaac 相機轉為 ROS 光學座標
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.5, -0.5, 0.5, -0.5), 
            convention="ros"
        ),
    )
    
    camera = Camera(cfg=camera_cfg)

    # 6. 開始模擬
    sim.reset()
    print("Simulation started...")

    # 讓模擬跑幾步，讓物理穩定且機器人落到地面
    for _ in range(50):
        sim.step()
    
    # 更新相機數據
    camera.update(dt=sim.get_physics_dt())

    # 7. 獲取數據並儲存
    print("Capturing data...")
    rgb = camera.data.output["rgb"][0].cpu().numpy() # [H, W, 3]
    depth = camera.data.output["distance_to_image_plane"][0].cpu().numpy() # [H, W, 1]
    intrinsic = camera.data.intrinsic_matrices[0].cpu().numpy()

    # 處理 RGB (去除 Alpha 通道)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    # 生成點雲
    pcd = create_point_cloud(rgb, depth, intrinsic)
    
    # (選用) 轉換到世界座標
    # 如果您希望點雲座標是相對於 "世界" 而不是 "相機"
    # camera_pose = camera.data.pos_w[0] ... (需構建 4x4 矩陣)
    # pcd.transform(camera_pose_matrix)

    # 存檔
    output_filename = "jetrover_scan.ply"
    o3d.io.write_point_cloud(output_filename, pcd)
    print(f"Point cloud saved to {output_filename}")

    simulation_app.close()

if __name__ == "__main__":
    main()