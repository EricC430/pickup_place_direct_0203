import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive Home Pose Tuning Tool")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.viewport.utility
from isaaclab.sim import SimulationContext, SimulationCfg, GroundPlaneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils
import math
import torch

from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.jetrover import JETROVER_CFG

def main():
    print("[INFO] Starting standalone interactive simulation for tuning home pose...")
    
    # 1. Initialize simulation context on CPU to allow Physics Inspector interaction
    sim_cfg = SimulationCfg(dt=0.01, device="cpu")
    sim = SimulationContext(sim_cfg)

    # 2. Spawn Ground
    cfg_ground = GroundPlaneCfg()
    cfg_ground.func("/World/Ground", cfg_ground)

    # 3. Spawn Robot using the same configuration as the environment
    # Note: We use the asset path defined in JETROVER_CFG
    robot_usd_path = JETROVER_CFG.spawn.usd_path
    cfg_robot = UsdFileCfg(usd_path=robot_usd_path)
    cfg_robot.func("/World/Robot", cfg_robot, translation=(0.0, 0.0, 0.0))

    # 4. Spawn a sample target object
    object_usd_path = "/workspace/test_isaaclab/ObjectFolder_selected/39/39.usd"
    cfg_obj = UsdFileCfg(usd_path=object_usd_path, scale=(0.6, 0.6, 0.6))
    cfg_obj.func("/World/TargetObject", cfg_obj, translation=(0.14875, 0.0, 0.05)) # Approximate center of table

    # 5. Spawn Camera (High-Res equivalent) attached to the robot wrist
    camera_prim_path = "/World/Robot/base_footprint/depth_cam_link/camera_mount_marker/Camera_High"
    target_hfov = 79.0
    aperture = 20.955
    calc_focal_length = aperture / (2 * math.tan(math.radians(target_hfov / 2)))

    camera_cfg = CameraCfg(
        prim_path=camera_prim_path,
        update_period=0.0,
        height=400, 
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=calc_focal_length, 
            horizontal_aperture=aperture,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.5, -0.5, 0.5, -0.5), # Assuming ROS convention used in env
            convention="ros"
        ),
    )
    
    # In Isaac Lab, sensors strictly need the physics engine up and running.
    # We will instantiate the sensor instance later after step.
    
    camera = Camera(cfg=camera_cfg)

    # 6. Reset simulation
    sim.reset()
    
    print("=" * 80)
    print("🎯 NATIVE INTERACTIVE PHYSICS MODE ENABLED 🎯")
    print("1. Physics is running on the CPU, allowing full GUI interaction.")
    print("2. You can use the 'Physics Inspector' (Window -> Physics -> Physics Inspector)")
    print("   to manually drag the robot's joints or select the Articulation Root to view joint values.")
    print("3. Switch your Perspective Camera to '/World/Robot/base_footprint/depth_cam.../Camera_High'")
    print("   to see exactly what the robot sees as you move its joints.")
    print("4. Read the target joint angles you visually found and write them back into:")
    print("   pickup_place_vision_asym_0310_env_cfg.py -> arm_init_offset_range")
    print("=" * 80)

    try:
        while simulation_app.is_running():
            sim.step()
            camera.update(sim.get_physics_dt())
    except KeyboardInterrupt:
        print("Exiting...")

    simulation_app.close()

if __name__ == "__main__":
    main()
