import torch
import argparse
import os

def deep_inspect(checkpoint_path):
    print(f"\n[INFO] Deep Inspecting checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] File not found: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 1. Inspect model_state_dict
    if "model_state_dict" in checkpoint:
        print("\n--- model_state_dict Keys ---")
        state_dict = checkpoint["model_state_dict"]
        found_keys = []
        for k, v in state_dict.items():
            if any(x in k.lower() for x in ["std", "noise", "sigma", "log"]):
                found_keys.append(k)
                print(f"  - {k}: shape={v.shape}, mean={v.mean().item():.4f}, min={v.min().item():.4f}, max={v.max().item():.4f}")
        
        if not found_keys:
            print("  [!] No obvious noise keys found. Listing all keys:")
            for k in state_dict.keys():
                print(f"    - {k}")
    else:
        print("\n[WARN] No 'model_state_dict' found. Top-level keys were:")
        for k in checkpoint.keys():
            print(f"  - {k}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, help="Path to the .pt checkpoint")
    args = parser.parse_args()
    deep_inspect(args.path)
