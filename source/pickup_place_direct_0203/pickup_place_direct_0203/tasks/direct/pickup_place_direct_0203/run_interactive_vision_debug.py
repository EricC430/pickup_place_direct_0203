#!/usr/bin/env python3
"""
Simple launcher for interactive_vision_debug.py

This script properly initializes the Isaac Lab environment and runs the 
interactive vision debug visualization.

Usage:
    python run_interactive_vision_debug.py

Or to enable GPU physics (if available):
    python run_interactive_vision_debug.py --physics_gpu
"""

import subprocess
import sys
import os
from pathlib import Path


def setup_isaac_lab_env():
    """Setup Isaac Lab environment variables if not already set."""
    if "ISAACLAB_PATH" not in os.environ:
        # Try to find Isaac Lab installation
        isaac_lab_paths = [
            Path("/workspace/isaaclab"),
            Path.home() / ".local/isaaclab",
            Path("/opt/isaaclab"),
        ]
        
        for path in isaac_lab_paths:
            if path.exists():
                os.environ["ISAACLAB_PATH"] = str(path)
                print(f"[Setup] Found Isaac Lab at: {path}")
                break
    
    if "OMNI_KIT_ALLOW_ROOT" not in os.environ:
        os.environ["OMNI_KIT_ALLOW_ROOT"] = "1"
        print("[Setup] Enabled OMNI_KIT_ALLOW_ROOT=1")


def main():
    """Run the interactive vision debug script."""
    setup_isaac_lab_env()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    interactive_script = script_dir / "interactive_vision_debug.py"
    
    if not interactive_script.exists():
        print(f"ERROR: Could not find interactive_vision_debug.py")
        print(f"  Expected location: {interactive_script}")
        return 1
    
    print(f"[Launcher] Starting interactive vision debug...")
    print(f"[Launcher] Script: {interactive_script}")
    print(f"[Launcher] Isaac Lab: {os.environ.get('ISAACLAB_PATH', 'auto-detect')}")
    print()
    
    try:
        # Run the interactive vision debug script
        # Pass through any additional arguments from command line
        cmd = [sys.executable, str(interactive_script)] + sys.argv[1:]
        result = subprocess.run(cmd, cwd=str(script_dir))
        return result.returncode
    except KeyboardInterrupt:
        print("\n[Launcher] Interrupted by user")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to run script: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
