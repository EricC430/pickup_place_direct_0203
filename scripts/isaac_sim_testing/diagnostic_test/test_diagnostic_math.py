import torch
import numpy as np
import sys
import os

# Add relevant paths
sys.path.append("/home/eric/isaaclab_volume/pickup_place_direct_0203/source/pickup_place_direct_0203")
from pickup_place_direct_0203.tasks.direct.pickup_place_direct_0203.utils.diagnostic_utils import DiagnosticProbe

class MockCfg:
    def __init__(self):
        self.action_scale = 1.0 #0.05
        self.action_cfg = {
            "arm_offsets": [0.0] * 5,
            "arm_scale": 1.0,
            "gripper_scale": 1.0,
            "gripper_offset": 0.0
        }

class MockRobotData:
    def __init__(self, device):
        self.default_joint_pos = torch.zeros((1, 6), device=device)
        self.default_joint_vel = torch.zeros((1, 6), device=device)

class MockRobot:
    def __init__(self, device):
        self.data = MockRobotData(device)

class MockEnv:
    def __init__(self):
        self.device = "cpu"
        self.num_envs = 1
        self.cfg = MockCfg()
        self.robot = MockRobot(self.device)
        self._arm_joint_indices = [0, 1, 2, 3, 4]
        self._gripper_joint_idx = [5]
        
        # Buffers
        self.joint_pos = torch.zeros((1, 9), device=self.device) # Real robot might have more joints
        self.action_history_buf = torch.zeros((1, 4, 6), device=self.device)
        self.cnn_feature_history_buf = torch.zeros((1, 13, 128), device=self.device)
        
        # Camera
        class MockCameraData:
            def __init__(self): self.output = {"rgb": torch.zeros((1, 80, 128, 3)), "depth": torch.zeros((1, 80, 128, 1))}
        class MockCamera:
            def __init__(self): self.data = MockCameraData()
        self.camera_low = MockCamera()
        
        self.common_step_counter = 500

def test_jerr():
    print("Testing JErr logic...")
    env = MockEnv()
    probe = DiagnosticProbe(env, log_dir="/tmp/diag_test")
    
    # Simulate a target: Move joint 0 to 1.0 rad
    target_action = torch.zeros((1, 6))
    target_action[0, 0] = 1.0 # action=1.0 -> target=1.0
    env.action_history_buf[0, -1, :] = target_action
    
    # Simulate current pos: robot is at 0.8 rad
    env.joint_pos[0, 0] = 0.8
    
    # Run verification
    probe._verify_jerr(0, 500)
    
    # In logic: target_rel (1.0) - jpos_rel (0.8) = 0.2
    # Check if results show correctness or if math matches
    print("JErr test passed if output shows JErr: [0.2 0. 0. 0. 0. 0.]")

def test_buffer_order():
    print("\nTesting Buffer Order logic...")
    env = MockEnv()
    probe = DiagnosticProbe(env, log_dir="/tmp/diag_test")
    
    # Fill buffer with increasing values
    # Index 0 is oldest, Index 12 is newest
    for i in range(13):
        env.cnn_feature_history_buf[0, i, :] = i
        
    probe.buffer_verifier.verify(0, 500)
    # MAD between frames [0, 4, 8, 12] should be 4.0

def test_pointnet_filter():
    print("\nTesting PointNet filter logic...")
    env = MockEnv()
    # Mock depth image with a gradient
    env.camera_low.data.output["depth"][0, :, :, 0] = 0.5
    
    # Manually set current_ptcloud
    env.current_ptcloud = torch.zeros((1, 1024, 3))
    env.current_ptcloud[0, :, 2] = 0.5 # Centroid at 0.5 depth
    
    probe = DiagnosticProbe(env, log_dir="/tmp/diag_test")
    probe.pointnet_probe.verify(0, 500)

if __name__ == "__main__":
    test_jerr()
    test_buffer_order()
    test_pointnet_filter()
