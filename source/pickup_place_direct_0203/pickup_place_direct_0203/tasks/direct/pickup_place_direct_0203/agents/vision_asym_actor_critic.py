# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import os
from torch.distributions import Normal
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from ..vision_encoder import ResNetVisonEncoder, PointNetEncoder

class VisionAsymActorCritic(ActorCritic):
    """
    Modular Actor-Critic for RSL-RL v3.0.1.
    Inherits from the monolithic ActorCritic but overrides observation extraction
    to support TensorDict inputs (RGBD images + PointClouds + Proprioception).
    """
    def __init__(self, obs, obs_groups, num_actions, **kwargs):
        # We must bypass the base ActorCritic.__init__'s strict 1D observation assertion
        # instead of calling super().__init__ which would crash on TensorDict observations.
        nn.Module.__init__(self)
        
        self.obs_groups = obs_groups
        # Architecture-specific latent dimensions
        num_actor_obs = 1130  # Proprio(42) + ResNet(512) + PointNet(512) + HighRes(64)
        # Dynamically infer critic dimension from observation dict (supports 0318=73, 0403=89, etc.)
        if isinstance(obs, dict) and "critic" in obs and hasattr(obs["critic"], "shape"):
            num_critic_obs = obs["critic"].shape[-1]
        else:
            num_critic_obs = kwargs.get("num_critic_obs", 73)
        print(f"[VisionAsymActorCritic] num_critic_obs = {num_critic_obs}")
        
        # Standard AC parameters from config
        actor_hidden_dims = kwargs.get("actor_hidden_dims", [512, 256, 128])
        critic_hidden_dims = kwargs.get("critic_hidden_dims", [512, 256, 128])
        activation = kwargs.get("activation", "elu")
        init_noise_std = kwargs.get("init_noise_std", 1.0)
        
        # ========== MODULES SEPARATE INITIALIZATION ==========
        # 1. Vision Encoders
        self.resnet_encoder = ResNetVisonEncoder(80, 128, 128, pretrained=True)
        self.pointnet_encoder = PointNetEncoder(out_dim=128)
        
        # 2. Base Actor/Critic MLPs (Using name mapping for v3.0.1 compatibility)
        from rsl_rl.networks import MLP, EmpiricalNormalization
        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        
        # 3. Observation Normalization
        self.actor_obs_normalization = kwargs.get("actor_obs_normalization", False)
        self.critic_obs_normalization = kwargs.get("critic_obs_normalization", False)
        
        if self.actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
            
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()
            
        # 4. Action Noise (Matching Base Logic)
        self.noise_std_type = kwargs.get("noise_std_type", "scalar")
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        else:
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            
        # 5. Stability (Added 0321 due to NaN crash)
        self.actor_input_ln = nn.LayerNorm(num_actor_obs)
            
        self.distribution = None
        print(f"[VisionAsymActorCritic] v3.0.1 Monolithic Init. Actor-Dim: {num_actor_obs}, Critic-Dim: {num_critic_obs}")


    def get_actor_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Returns the observations for the actor, extracting features if modular."""
        if isinstance(obs, torch.Tensor):
            return obs
            
        # [DEBUG] One-time structure verification
        if not hasattr(self, "_actor_obs_debug_done"):
            print(f"🚀 [DEBUG] get_actor_obs - Input type: {type(obs)}")
            if hasattr(obs, 'keys'): print(f"🚀 [DEBUG] get_actor_obs - Keys: {obs.keys()}")
            self._actor_obs_debug_done = True
            
        # Check if we have the specific modular keys (from flat obs_groups)
        if "policy_proprio" in obs.keys():
            return self._extract_features(obs)
            
        # Backward compatibility for 'policy' key nested structure
        if "policy" in obs.keys():
            policy_content = obs["policy"]
            if isinstance(policy_content, torch.Tensor):
                return policy_content
            # If it's a nested dict, we handle it
            return self._extract_features(policy_content)
                
        # Default to base logic (concatenate everything in the dict)
        if hasattr(self, "obs_groups") and "policy" in self.obs_groups:
             return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)
             
        return obs

    def get_critic_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """Returns the observations for the critic (privileged info)."""
        if isinstance(obs, torch.Tensor):
            return obs
        # Critic group is typically already a flat tensor
        if "critic" in obs.keys():
            return obs["critic"]
        return super().get_critic_obs(obs)

    def _extract_features(self, obs_dict: TensorDict | dict) -> torch.Tensor:
        """Processes vision dict into a 1130-dim feature vector.
        
        CRITICAL: This method must NEVER return NaN/Inf values.
        The safety chain is:
        1. Sanitize encoder inputs (replace NaN with 0)
        2. Run encoders
        3. Concatenate features
        4. Apply LayerNorm
        5. Final nan_to_num AFTER LayerNorm (catches LayerNorm-on-constant edge case)
        """
        # Supports both flat keys (policy_proprio) and nested keys (proprio)
        proprio = obs_dict.get("policy_proprio", obs_dict.get("proprio"))
        images = obs_dict.get("policy_images", obs_dict.get("images"))     # (B, 4, 3, 80, 128)
        points = obs_dict.get("policy_points", obs_dict.get("points"))      # (B, 4, 1024, 3)
        high_res = obs_dict.get("policy_high_res", obs_dict.get("high_res")) # (B, 64)

        B = proprio.shape[0]

        # PRE-SANITIZE: Ensure encoder inputs are finite (handles uninitialized buffers)
        images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)
        points = torch.nan_to_num(points, nan=0.0, posinf=1.0, neginf=-1.0)
        proprio = torch.nan_to_num(proprio, nan=0.0, posinf=1.0, neginf=-1.0)
        high_res = torch.nan_to_num(high_res, nan=0.0, posinf=1.0, neginf=-1.0)

        # 1. Process Low-Res Images (Time-aware stacking)
        # ResNet18: (B, 4, 3, 80, 128) -> (B*4, 3, 80, 128) -> (B*4, 128) -> (B, 512)
        res_feat = self.resnet_encoder(images.reshape(B*4, 3, 80, 128))
        res_feat = res_feat.view(B, 512)
        
        # 2. Process Point Clouds
        # PointNet: (B, 4, 1024, 3) -> (B*4, 1024, 3) -> (B*4, 128) -> (B, 512)
        pt_feat = self.pointnet_encoder(points.reshape(B*4, 1024, 3))
        pt_feat = pt_feat.view(B, 512)
        
        # POST-SANITIZE encoder outputs (catches encoder internal NaN from random init)
        res_feat = torch.nan_to_num(res_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        pt_feat = torch.nan_to_num(pt_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 3. Concatenate all: [Proprio(42) | ResNet(512) | PointNet(512) | HighRes(64)] = 1130
        out = torch.cat([proprio, res_feat, pt_feat, high_res], dim=-1)
        
        # 4. Apply LayerNorm for numerical stability across modular features
        out = self.actor_input_ln(out)
        
        # 5. FINAL safety: LayerNorm on a constant vector (e.g. all zeros) can produce NaN
        #    because it divides by std which is 0. This is the LAST line of defense.
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
            
        return out

    def update_distribution(self, obs):
        """Override base update_distribution with NaN-safe version.
        
        The base class calls Normal(mean, std) which validates args and throws
        ValueError if mean contains NaN. We intercept and sanitize mean BEFORE
        constructing the distribution.
        """
        # Compute mean from actor MLP
        mean = self.actor(obs)
       
        # CRITICAL: Sanitize mean BEFORE Normal() validation
        if torch.isnan(mean).any() or torch.isinf(mean).any():
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        
        # Sanitize std as well (must be positive)
        # [RESTORE] 恢復為全局統一的探索雜訊下限
        std = torch.clamp(std, min=1e-6)
        if torch.isnan(std).any() or torch.isinf(std).any():
            # 如果 Checkpoint 損壞，則使用 cfg 中設定的 init noise (1.0) 作為備用值
            std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=0.01)
            std = torch.clamp(std, min=1e-6)
        
        # Create distribution (now guaranteed valid)
        self.distribution = Normal(mean, std)

    def update_normalization(self, obs: TensorDict | torch.Tensor):
        """Explicitly normalize based on extracted features only."""
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs)
        if self.critic_obs_normalization:
            critic_obs = self.get_critic_obs(obs)
            self.critic_obs_normalizer.update(critic_obs)

    # def load_state_dict(self, state_dict, strict=True):
    #     """Custom loader to map monolithic RSL-RL keys to modular architecture.
    #     Detects if checkpoint is already modular to avoid double-remapping.
    #     """
    #     # Detect modularity: check if 'actor.layers.0.weight' or 'critic.layers.0.weight' exists
    #     is_already_modular = False
    #     for key in state_dict.keys():
    #         if "actor.layers." in key or "critic.layers." in key:
    #             is_already_modular = True
    #             break
    #     
    #     if is_already_modular:
    #         print(f"[VisionAsymActorCritic] Detected modular checkpoint. Skipping remapping.")
    #         return super().load_state_dict(state_dict, strict=False)
    #
    #     new_state_dict = {}
    #     for key, value in state_dict.items():
    #         new_key = key
    #         # 1. Map Actor MLP layers: actor.0 -> actor.layers.0
    #         if key.startswith("actor.") and not key.startswith("actor.layers."):
    #             parts = key.split(".")
    #             if parts[1].isdigit():
    #                 new_key = f"actor.layers.{'.'.join(parts[1:])}"
    #         
    #         # 2. Map Critic MLP layers: critic.0 -> critic.layers.0
    #         elif key.startswith("critic.") and not key.startswith("critic.layers."):
    #             parts = key.split(".")
    #             if parts[1].isdigit():
    #                 new_key = f"critic.layers.{'.'.join(parts[1:])}"
    #         
    #         # 3. Handle std key differences
    #         elif key == "actor_critic.std" or key == "std":
    #             new_key = "std"
    #         elif key == "actor_critic.log_std" or key == "log_std":
    #             new_key = "log_std"
    #
    #         # 4. Handle Normalizer keys (ensure direct mapping if from monolithic)
    #         elif "actor_obs_normalizer" in key or "critic_obs_normalizer" in key:
    #             value = torch.nan_to_num(value, nan=0.0)
    #         
    #         new_state_dict[new_key] = value
    #
    #     print(f"[VisionAsymActorCritic] Monolithic-to-Modular: remapped {len(new_state_dict)} keys.")
    #     return super().load_state_dict(new_state_dict, strict=False)

    def act(self, obs: TensorDict | torch.Tensor, **kwargs) -> torch.Tensor:
        """Overrides act to ensure numerical safety in the distribution."""
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        
        # Safety: ensure actor_obs is finite before MLP
        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        """Overrides act_inference to ensure numerical safety."""
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        actions = self.actor(obs)
        return torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)

    def evaluate(self, obs: TensorDict | torch.Tensor, **kwargs) -> torch.Tensor:
        """Returns value estimates with numerical safety."""
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        
        # NaN Protection for Critic Input
        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
        value = self.critic(critic_obs)
        
        # Final value safety
        value = torch.nan_to_num(value, nan=0.0, posinf=100.0, neginf=-100.0)
            
        return value

    @property
    def entropy(self):
        """Returns sanitized entropy of the action distribution."""
        if self.distribution is None:
            return torch.tensor(0.0, device=self.std.device)
        entropy = self.distribution.entropy().sum(dim=-1)
        return torch.nan_to_num(entropy, nan=0.0, posinf=1.0, neginf=-1.0)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Returns sanitized log probabilities of actions."""
        if self.distribution is None:
            return torch.tensor(-10.0, device=actions.device)
        log_prob = self.distribution.log_prob(actions).sum(dim=-1)
        # Avoid extremely low log_probs that can lead to NaN gradients
        return torch.nan_to_num(log_prob, nan=-10.0, posinf=1.0, neginf=-100.0)

    def load_modules(self, resnet_path=None, pointnet_path=None, mlp_path=None):
        """
        Loads pre-trained weights for specific sub-modules.
        """
        if resnet_path and os.path.exists(resnet_path):
            print(f"[VisionAsymActorCritic] Loading ResNet from {resnet_path}")
            device = next(self.resnet_encoder.parameters()).device
            self.resnet_encoder.load_state_dict(torch.load(resnet_path, map_location=device))
        
        if pointnet_path and os.path.exists(pointnet_path):
            print(f"[VisionAsymActorCritic] Loading PointNet from {pointnet_path}")
            device = next(self.pointnet_encoder.parameters()).device
            self.pointnet_encoder.load_state_dict(torch.load(pointnet_path, map_location=device))
            
        if mlp_path and os.path.exists(mlp_path):
            print(f"[VisionAsymActorCritic] Loading MLP state from {mlp_path}")
            device = next(self.actor.parameters()).device
            # Loads entire state dict (actor, critic, std)
            self.load_state_dict(torch.load(mlp_path, map_location=device), strict=False)
