import argparse
import os
import torch
import numpy as np
from PIL import Image, ImageDraw

# IsaacLab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isolate and Verify Camera Extrinsics")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Let's import the specific task class
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0403_env import PickupPlaceVisionAsym0403Env
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0403_env_cfg import PickupPlaceVisionAsym0403EnvCfg

def main():
    print("[VerifyExtrinsics] Starting isolated geometric verification.")
    
    # 1. Setup Environment
    env_cfg = PickupPlaceVisionAsym0403EnvCfg()
    env_cfg.scene.num_envs = 1
    
    env = PickupPlaceVisionAsym0403Env(cfg=env_cfg)
    
    # Reset and warm up rendering
    env.reset()
    for _ in range(5):
        env.sim.render()
        env.sim.step()
    
    # Get camera data
    camera = env.scene["camera_high"]
    camera.update(dt=0.0)
    
    cam_pos_w = camera.data.pos_w[0]
    cam_quat_w_raw_gl = camera.data.quat_w_opengl[0] # The raw USD orientation
    K_np = camera.data.intrinsic_matrices[0].cpu().numpy()
    
    device = env.device

    # =========================================================================
    # STEP 1: SOLVE FOR ACCURATE ROTATION MATRIX
    # =========================================================================
    # Following the solver logic to find the best Right-Handed basis
    cam_rot_mat_gl_P3D = matrix_from_quat(cam_quat_w_raw_gl.unsqueeze(0)).squeeze(0)
    cam_rot_mat_gl = cam_rot_mat_gl_P3D.T  # P3D -> Column Vector conversion

    obj_pos_w = env.scene["object"].data.root_pos_w[0]
    vec = obj_pos_w - cam_pos_w
    
    import itertools
    best_R = None
    best_score = -float('inf')
    axes = [cam_rot_mat_gl[:, 0], cam_rot_mat_gl[:, 1], cam_rot_mat_gl[:, 2]]
    
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            R = torch.stack([signs[0]*axes[perm[0]], signs[1]*axes[perm[1]], signs[2]*axes[perm[2]]], dim=1)
            if torch.linalg.det(R) < 0: continue
            
            # Target signs: x < 0 (Left), y > 0 (Down), z > 0 (Forward)
            px = torch.dot(vec, R[:, 0]).item()
            py = torch.dot(vec, R[:, 1]).item()
            pz = torch.dot(vec, R[:, 2]).item()
            
            score = (1.0 if px < 0 else -1.0) * abs(px) + (1.0 if py > 0 else -1.0) * abs(py) + (1.0 if pz > 0 else -1.0) * abs(pz)
            if score > best_score:
                best_score = score
                best_R = R

    cam_rot_mat_ros = best_R
    cam_T_world = torch.eye(4, device=device)
    cam_T_world[:3, :3] = cam_rot_mat_ros
    cam_T_world[:3, 3] = cam_pos_w
    
    # =========================================================================
    # STEP 2: TEST POINTS
    # =========================================================================
    pts_cam = torch.tensor([
        [-0.1, -0.1, 0.4], # Top-Left
        [ 0.1, -0.1, 0.4], # Top-Right
        [-0.1,  0.1, 0.4], # Bottom-Left
        [ 0.1,  0.1, 0.4]  # Bottom-Right
    ], device=device)
    pts_cam_h = torch.cat([pts_cam, torch.ones(4, 1, device=device)], dim=1)
    pts_world = (cam_T_world @ pts_cam_h.T).T[:, :3]

    # =========================================================================
    # STEP 3: VISUAL MARKERS
    # =========================================================================
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/VerificationMarkers",
        markers={
            "sphere": sim_utils.SphereCfg(radius=0.015, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)))
        }
    )
    markers = VisualizationMarkers(marker_cfg)
    markers.visualize(translations=pts_world)
    
    for _ in range(5):
        env.sim.render()
        env.sim.step()
    
    camera.update(dt=0.0)
    
    # =========================================================================
    # STEP 4: PROJECTION SAVE
    # =========================================================================
    rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    
    fx, fy = K_np[0, 0], K_np[1, 1]
    cx, cy = K_np[0, 2], K_np[1, 2]
    
    for pt in pts_cam:
        u = (fx * pt[0].item() / pt[2].item()) + cx
        v = (fy * pt[1].item() / pt[2].item()) + cy
        draw.ellipse((u-10, v-10, u+10, v+10), outline="red", width=2)
        draw.text((u+15, v-5), "MATH", fill="red")
        
    out_dir = "/workspace/test_isaaclab/pickup_place_direct_0203/logs/verify_extrinsics"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "projection_test.png")
    img.save(out_path)
    print(f"[VerifyExtrinsics] Saved isolated verification image to: {out_path}")
    
    simulation_app.close()

if __name__ == "__main__":
    main()
