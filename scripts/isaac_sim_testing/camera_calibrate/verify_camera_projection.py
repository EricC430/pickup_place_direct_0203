import os
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw

def create_isolation_test_image(gt_coord_w, cam_pos_w, cam_quat_w_usd, intrinsic_matrix, rgb_tensor, output_path):
    import isaaclab.utils.math as math_utils
    # 1. Pure Math Projection Setup
    device = cam_pos_w.device
    cam_rot_mat = math_utils.matrix_from_quat(cam_quat_w_usd.unsqueeze(0)).squeeze(0)
    
    # ── Test matrix derivation without column flip ──
    cam_T_world = torch.eye(4, device=device, dtype=torch.float32)
    cam_T_world[:3, :3] = cam_rot_mat
    cam_T_world[:3, 3] = cam_pos_w
    cam_T_world_inv = torch.inverse(cam_T_world)

    # 2. Transform the World Point to Camera Coordinates
    pt_w_h = torch.cat([gt_coord_w, torch.ones(1, device=device)])
    pt_cam = (cam_T_world_inv @ pt_w_h)[:3]

    print(f"[Math Isolation] World coord: {gt_coord_w.cpu().numpy()}")
    print(f"[Math Isolation] Extracted Camera coord: {pt_cam.cpu().numpy()}")

    # 3. Project to pixels
    rgb_np = rgb_tensor.cpu().numpy().copy()
    if rgb_np.shape[2] == 4:
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGBA2RGB)
    
    pil_img = Image.fromarray(rgb_np)
    draw = ImageDraw.Draw(pil_img)

    fx = intrinsic_matrix[0, 0].item()
    fy = intrinsic_matrix[1, 1].item()
    cx = intrinsic_matrix[0, 2].item()
    cy = intrinsic_matrix[1, 2].item()

    if pt_cam[2] > 0:
        u = (fx * pt_cam[0] / pt_cam[2]) + cx
        v = (fy * pt_cam[1] / pt_cam[2]) + cy
        print(f"[Math Isolation] Projected Pixels: u={u:.2f}, v={v:.2f}")
        draw.ellipse((u-5, v-5, u+5, v+5), outline="lime", fill="red")
        draw.text((u+8, v-8), "Math Projection", fill="lime")
    else:
        print("[Math Isolation] ERROR: Projected point has negative Z depth (is behind camera)!")

    pil_img.save(output_path)
    print(f"[Math Isolation] Saved to {output_path}")
