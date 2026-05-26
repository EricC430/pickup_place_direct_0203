import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import csv
import os

class DiagnosticProbe:
    def __init__(self, env, log_dir="diagnostic_logs"):
        self.env = env
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DiagnosticProbe] Initialized. Logging to {self.log_dir.absolute()}")
        
        self.sync_verifier = SyncVerifier(self)
        self.buffer_verifier = BufferVerifier(self)
        self.pointnet_probe = PointNetProbe(self)
        self.feature_probe = FeatureProbe(self)
        
        self.results = {}

    def update(self, obs_dict, step_count):
        env_id = 0
        if step_count % 50 == 0 or step_count == 1:
            print(f"[DiagnosticProbe] Running verification at step {step_count}...")
            self.sync_verifier.verify(env_id, step_count)
            self.buffer_verifier.verify(env_id, step_count)
            self.pointnet_probe.verify(env_id, step_count)
            self.feature_probe.log_and_verify(obs_dict, env_id, step_count)
            self._verify_jerr(env_id, step_count)
            # Only verify assembly if it's the concatenated version
            if "policy" in obs_dict and isinstance(obs_dict["policy"], torch.Tensor):
                self._verify_obs_assembly(obs_dict, step_count)
            else:
                print(f"  [Obs Assembly Check] Skipped (Raw Mode: keys={list(obs_dict.keys())})")

    def _verify_jerr(self, env_id, step_count):
        # jerr = target_jpos_relative - current_jpos_relative
        # current_jpos is (B, 6) relative to default
        # target_jpos is derived from actions
        
        # We need to access the internal tensors of the env
        if not hasattr(self.env, "joint_pos"): return
        
        # This matches the logic in _get_observations
        indices = list(self.env._arm_joint_indices) + list(self.env._gripper_joint_idx)
        current_jpos_full = self.env.joint_pos[env_id, indices]
        default_jpos_full = self.env.robot.data.default_joint_pos[env_id, indices]
        
        # Relative JPos (what is used in obs)
        jpos_rel = current_jpos_full - default_jpos_full
        
        # Last action scaling logic
        prev_actions = self.env.action_history_buf[env_id, -1, :]
        scaled_actions = prev_actions * self.env.cfg.action_scale
        
        arm_offsets = torch.tensor(self.env.cfg.action_cfg["arm_offsets"], device=self.env.device)
        arm_scale = self.env.cfg.action_cfg["arm_scale"]
        prev_arm_targets = scaled_actions[:5] * arm_scale + arm_offsets
        # [0402 Numerical Safety] Sync diagnostic logic with physical clipping in env
        prev_arm_targets = torch.clamp(prev_arm_targets, min=-2.09, max=2.09)

        gripper_scale = self.env.cfg.action_cfg["gripper_scale"]
        gripper_offset = self.env.cfg.action_cfg["gripper_offset"]
        prev_gripper_target = scaled_actions[5] * gripper_scale + gripper_offset
        # [0402 Numerical Safety] Sync diagnostic logic with physical clipping in env
        prev_gripper_target = torch.clamp(prev_gripper_target, min=0.0, max=1.57)
        
        prev_target_full = torch.cat([prev_arm_targets, prev_gripper_target.unsqueeze(0)])
        prev_target_rel = prev_target_full - default_jpos_full
        
        jerr = prev_target_rel - jpos_rel
        
        print(f"\n[JErr Check @ Step {step_count}]")
        print(f"  Target Rel: {prev_target_rel.cpu().numpy()}")
        print(f"  JPos Rel:   {jpos_rel.cpu().numpy()}")
        print(f"  JErr:       {jerr.cpu().numpy()}")
        
        if jerr.abs().max() > 3.14:
            print(f"  ⚠️ [WARNING] Action Explosion detected! Max JErr: {jerr.abs().max().item():.2f} rad (Check action_scale or normalization)")
        
        self.results["jerr_correctness"] = "PASS" if jerr.abs().max() < 3.14 else "FAIL (EXPLOSION)"

    def _verify_obs_assembly(self, obs_dict, step_count):
        print(f"\n[Obs Assembly Check @ Step {step_count}]")
        key = "policy"
        if key in obs_dict:
            total_dim = obs_dict[key].shape[-1]
            print(f"  Policy Obs Dim: {total_dim}")
            self.results["obs_dimension_match"] = "PASS" if total_dim == 1130 else "FAIL (DIM MISMATCH)"
            
            # Check for NaNs/Infs
            if torch.isnan(obs_dict[key]).any():
                print("  ❌ NaNs detected in Policy Obs!")
                self.results["obs_dimension_match"] = "FAIL (NaNs)"
        
    def generate_report(self):
        print("\n" + "="*60)
        print("INPUT VALIDATION REPORT")
        print("="*60)
        checks = [
            "render_sync", "frame_buffer_order", "depth_valid_ratio",
            "pointcloud_centroid", "pointcloud_density", "resnet_temporal_variance",
            "resnet_spatial_sensitivity", "jerr_correctness", "obs_dimension_match"
        ]
        for check in checks:
            result = self.results.get(check, "NOT_RUN")
            icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌", "NOT_RUN": "⬜"}.get(result)
            print(f"{icon} {check:<45} [{result}]")
        print("="*60)

class SyncVerifier:
    def __init__(self, probe):
        self.probe = probe
        
    def verify(self, env_id, step_count):
        try:
            # 1. Physical state
            obj_pos = self.probe.env.scene["object"].data.root_pos_w[env_id]
            
            # 2. Image buffer
            if hasattr(self.probe.env, "camera_low"):
                img = self.probe.env.camera_low.data.output["rgb"][env_id]
                if img.shape[-1] == 4: img = img[..., :3]
                img_np = img.cpu().numpy().astype(np.uint8)
                
                # Check for black screen
                max_val = img_np.max()
                std_val = img_np.std()
                
                if max_val < 10:
                    self.probe.results["render_sync"] = "FAIL (BLACK)"
                elif std_val < 5:
                    self.probe.results["render_sync"] = "WARN (UNIFORM)"
                else:
                    self.probe.results["render_sync"] = "PASS"
                
                # Save snapshot
                Image.fromarray(img_np).save(self.probe.log_dir / f"sync_step_{step_count}.png")
                print(f"[SyncVerifier] Saved snapshot to sync_step_{step_count}.png (Max: {max_val}, Std: {std_val:.2f})")
        except Exception as e:
            print(f"[SyncVerifier] Error: {e}")

class BufferVerifier:
    def __init__(self, probe):
        self.probe = probe
        
    def verify(self, env_id, step_count):
        # Strided indices: 0, 4, 8, 12
        indices = [0, 4, 8, 12]
        
        # In raw mode, use raw_image_history_buf
        if getattr(self.probe.env.cfg, "use_raw_observations", False):
            if not hasattr(self.probe.env, "raw_image_history_buf"): return
            buffer = self.probe.env.raw_image_history_buf[env_id, indices, ...]
            mode = "RAW_IMAGE"
        else:
            if not hasattr(self.probe.env, "cnn_feature_history_buf"): return
            buffer = self.probe.env.cnn_feature_history_buf[env_id, indices, :] # (4, 128)
            mode = "CNN_FEAT"
        
        print(f"\n[BufferVerifier ({mode}) @ Step {step_count}]")
        all_diffs = []
        for i in range(3):
            diff = (buffer[i] - buffer[i+1]).abs().mean().item()
            all_diffs.append(diff)
            print(f"  Frame {i} (older) vs {i+1} (newer) MAD: {diff:.6f}")
        
        if all(d > 1e-6 for d in all_diffs):
            self.probe.results["frame_buffer_order"] = "PASS"
        else:
            self.probe.results["frame_buffer_order"] = "FAIL (STATIC)"

class PointNetProbe:
    def __init__(self, probe):
        self.probe = probe
        
    def verify(self, env_id, step_count):
        if not hasattr(self.probe.env, "camera_low"): return
        
        # Depth check
        depth = self.probe.env.camera_low.data.output.get("depth")
        if depth is None:
            depth = self.probe.env.camera_low.data.output.get("distance_to_image_plane")
        
        if depth is not None:
            d = depth[env_id].squeeze(-1)
            valid_mask = (d > 0.1) & (d < 1.5)
            valid_count = valid_mask.sum().item()
            print(f"\n[PointNetProbe @ Step {step_count}]")
            print(f"  Valid Depth Pixels: {valid_count}")
            self.probe.results["depth_valid_ratio"] = "PASS" if valid_count > 500 else "WARN (EMPTY)"
            
        # PointCloud check
        if hasattr(self.probe.env, "current_ptcloud"):
            pts = self.probe.env.current_ptcloud[env_id] # (1024, 3)
            centroid = pts.mean(dim=0).cpu().numpy()
            bbox_min = pts.min(dim=0)[0].cpu().numpy()
            bbox_max = pts.max(dim=0)[0].cpu().numpy()
            vol = (bbox_max - bbox_min).prod()
            
            print(f"  PtCloud Centroid: {centroid}")
            print(f"  PtCloud BBox Vol: {vol:.6f} m^3")
            
            # Centroid check (Camera frame: Z is depth)
            if centroid[2] > 0.1 and centroid[2] < 1.0:
                self.probe.results["pointcloud_centroid"] = "PASS"
            else:
                self.probe.results["pointcloud_centroid"] = "FAIL (OUTSIDE)"
                
            self.probe.results["pointcloud_density"] = "PASS" if vol > 0.001 else "WARN (SQUEEZED)"
            
            # Save CSV
            np.savetxt(self.probe.log_dir / f"ptcloud_step_{step_count}.csv", pts.cpu().numpy(), delimiter=",", header="x,y,z", comments="")

class FeatureProbe:
    def __init__(self, probe):
        self.probe = probe
        self.history_feat = []
        self.history_pos = []
        
    def log_and_verify(self, obs_dict, env_id, step_count):
        # Extract features from normalized obs if available
        # Structure: [Proprio(42) | VisionLow(512) | PointNet(512) | VisionHigh(64)]
        if "policy" in obs_dict:
            policy_obs = obs_dict["policy"][env_id]
            vision_feat = policy_obs[42:42+512] # Vision Low features
        elif "policy_images" in obs_dict:
            # Under raw mode, we check variance of raw pixels instead of features
            img_buffer = obs_dict["policy_images"][env_id] # (4, 3, 80, 128)
            std_feat = img_buffer.std(dim=0).mean().item()
            print(f"\n[FeatureProbe (RAW_IMAGE) @ Step {step_count}]")
            print(f"  Mean Pixel Std (Temporal): {std_feat:.6f}")
            self.probe.results["resnet_temporal_variance"] = "PASS" if std_feat > 1e-4 else "FAIL (STATIC_IMG)"
            return
        else:
            return
        
        # EE position (Proxy for spatial change)
        if hasattr(self.probe.env, "ee_frame"):
            ee_pos = self.probe.env.ee_frame.data.target_pos_w[env_id, 0]
        else:
            ee_pos = torch.zeros(3, device=self.probe.env.device)
            
        self.history_feat.append(vision_feat.detach().clone())
        self.history_pos.append(ee_pos.detach().clone())
        
        if len(self.history_feat) < 10: return
        
        # 1. Temporal Identity Check
        recent_feats = torch.stack(self.history_feat[-10:])
        std_feat = recent_feats.std(dim=0).mean().item()
        print(f"\n[FeatureProbe @ Step {step_count}]")
        print(f"  Mean Feature Std (Temporal): {std_feat:.6f}")
        self.probe.results["resnet_temporal_variance"] = "PASS" if std_feat > 1e-4 else "FAIL (DEAD)"
        
        # 2. Spatial Sensitivity (Simplified: Distance correlation)
        if len(self.history_feat) > 20:
             feat_diff = (recent_feats[1:] - recent_feats[:-1]).norm(dim=1)
             pos_stack = torch.stack(self.history_pos[-10:])
             pos_diff = (pos_stack[1:] - pos_stack[:-1]).norm(dim=1)
             
             # Calculate correlation manually or just check if both are moving
             if pos_diff.mean() > 0.001: # Only if robot is moving
                 if feat_diff.mean() > 1e-4:
                     self.probe.results["resnet_spatial_sensitivity"] = "PASS"
                 else:
                     self.probe.results["resnet_spatial_sensitivity"] = "WARN (LOW SENSITIVITY)"
             else:
                 self.probe.results["resnet_spatial_sensitivity"] = "NOT_RUN (STATIC)"
