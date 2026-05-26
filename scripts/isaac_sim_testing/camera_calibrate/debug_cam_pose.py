
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.pickup_place_vision_asym_0403_env_cfg import PickupPlaceVisionAsym0403EnvCfg

def main():
    env_cfg = PickupPlaceVisionAsym0403EnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    
    # Get poses
    env_idx = 0
    cam_pos_w = env._camera.data.pos_w[env_idx]
    cam_quat_w = env._camera.data.quat_w_world[env_idx] # (w, x, y, z)
    obj_pos_w = env.scene["object"].data.root_pos_w[env_idx]
    
    print(f"DEBUG: Camera World Pos: {cam_pos_w.cpu().numpy()}")
    print(f"DEBUG: Camera World Quat (w,x,y,z): {cam_quat_w.cpu().numpy()}")
    print(f"DEBUG: Object World Pos: {obj_pos_w.cpu().numpy()}")
    
    # Calculate vector in World
    vec_w = obj_pos_w - cam_pos_w
    print(f"DEBUG: Vec World (Obj - Cam): {vec_w.cpu().numpy()}")
    
    # Check basis vectors from Quat
    from isaaclab.utils.math import matrix_from_quat
    R = matrix_from_quat(cam_quat_w.unsqueeze(0)).squeeze(0)
    # R cols are local axes in World. 
    # Isaac Camera convention: X-fwd?
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    print(f"DEBUG: Isaac Axis X (fwd?): {x_axis.cpu().numpy()} | Dot with Vec: {torch.dot(x_axis, vec_w).item()}")
    print(f"DEBUG: Isaac Axis Y (left?): {y_axis.cpu().numpy()} | Dot with Vec: {torch.dot(y_axis, vec_w).item()}")
    print(f"DEBUG: Isaac Axis Z (up?): {z_axis.cpu().numpy()} | Dot with Vec: {torch.dot(z_axis, vec_w).item()}")

    # Check OpenGL quat
    cam_quat_gl = env._camera.data.quat_w_opengl[env_idx]
    R_gl = matrix_from_quat(cam_quat_gl.unsqueeze(0)).squeeze(0)
    print(f"DEBUG: OpenGL Axis X (right?): {R_gl[:,0].cpu().numpy()} | Dot with Vec: {torch.dot(R_gl[:,0], vec_w).item()}")
    print(f"DEBUG: OpenGL Axis Y (up?): {R_gl[:,1].cpu().numpy()} | Dot with Vec: {torch.dot(R_gl[:,1], vec_w).item()}")
    print(f"DEBUG: OpenGL Axis Z (back?): {R_gl[:,2].cpu().numpy()} | Dot with Vec: {torch.dot(R_gl[:,2], vec_w).item()}")

    # Determine OpenCV mapping based on dots
    # OpenCV Z (forward) should have HIGHEST POSITIVE dot product with vec_w
    # OpenCV Y (down) should have... if camera is looking down, dot should be positive? 
    # If camera is looking down, Up axis (Z_i or Y_gl) dots will be negative.
    
    env.close()

if __name__ == "__main__":
    main()
