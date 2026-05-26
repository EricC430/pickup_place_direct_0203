import torch
import os
import argparse

def extract_vision_weights(bc_model_path, output_path):
    print(f"[INFO] Loading BC model from: {bc_model_path}")
    checkpoint = torch.load(bc_model_path, map_location="cpu")
    
    vision_weights = {}
    
    # 1. Vision Encoders
    # 2. LayerNorms
    # 3. EmpiricalNormalizer (proprio_norm)
    
    include_prefixes = [
        "vision_encoder_low.", 
        "pointnet.", 
        "vision_encoder_high.",
        "vision_low_ln.",
        "pointnet_ln.",
        "vision_high_ln.",
        "proprio_norm."
    ]
    
    keys_found = []
    for k, v in checkpoint.items():
        if any(k.startswith(p) for p in include_prefixes):
            vision_weights[k] = v
            keys_found.append(k)
            
    if not vision_weights:
        print("[ERROR] No vision or normalization weights found! Checking available keys:")
        print(list(checkpoint.keys())[:20])
        return

    print(f"[INFO] Extracted {len(vision_weights)} tensors (vision encoders + normalizers).")
    torch.save(vision_weights, output_path)
    print(f"[INFO] Saved extracted weights to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_model", type=str, required=True)
    parser.add_argument("--out", type=str, default="vision_weights_0318.pt")
    args = parser.parse_args()
    
    extract_vision_weights(args.bc_model, args.out)
