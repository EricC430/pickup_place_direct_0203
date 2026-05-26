# Copyright (c) 2026

"""Verification script for GR00T LeRobot v2 dataset.
   Validates folder structure, metadata json syntax, parquet columns, and video formats.
"""

import argparse
import os
import json
import glob
import pyarrow.parquet as pq

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def main():
    parser = argparse.ArgumentParser(description="Verify generated LeRobot v2 dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the LeRobot v2 dataset directory.")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    print(f"\n==================== Verifying Dataset: {dataset_dir} ====================")

    # 1. Check Directory Structure
    expected_paths = [
        "meta",
        "meta/info.json",
        "meta/episodes.jsonl",
        "meta/tasks.jsonl",
        "meta/modality.json",
        "data/chunk-000",
        "videos/chunk-000/observation.images.wrist"
    ]

    all_exist = True
    for rel_path in expected_paths:
        full_path = os.path.join(dataset_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"[FAIL] Missing expected path: {rel_path}")
            all_exist = False
        else:
            print(f"[OK] Found path: {rel_path}")

    if not all_exist:
        print("[ERROR] Dataset directory structure is incomplete. Exiting validation.")
        return

    # 2. Check JSON Metadata syntax
    try:
        with open(os.path.join(dataset_dir, "meta/info.json"), "r") as f:
            info = json.load(f)
        print("[OK] meta/info.json loaded successfully.")
        print(f"     - Codebase version: {info.get('codebase_version')}")
        print(f"     - FPS: {info.get('fps')}")
        print("     - Features:")
        for feat_name, feat_meta in info.get("features", {}).items():
            print(f"       * {feat_name}: dtype={feat_meta.get('dtype')}, shape={feat_meta.get('shape')}")
    except Exception as e:
        print(f"[FAIL] Failed to load meta/info.json: {e}")

    try:
        with open(os.path.join(dataset_dir, "meta/modality.json"), "r") as f:
            modality = json.load(f)
        print("[OK] meta/modality.json loaded successfully.")
    except Exception as e:
        print(f"[FAIL] Failed to load meta/modality.json: {e}")

    # Count tasks and episodes
    try:
        task_count = 0
        with open(os.path.join(dataset_dir, "meta/tasks.jsonl"), "r") as f:
            for line in f:
                if line.strip():
                    task_count += 1
        print(f"[OK] meta/tasks.jsonl read. Found {task_count} tasks.")
    except Exception as e:
        print(f"[FAIL] Failed to read meta/tasks.jsonl: {e}")

    try:
        episode_count = 0
        total_frames = 0
        with open(os.path.join(dataset_dir, "meta/episodes.jsonl"), "r") as f:
            for line in f:
                if line.strip():
                    ep = json.loads(line)
                    episode_count += 1
                    total_frames += ep.get("length", 0)
        print(f"[OK] meta/episodes.jsonl read. Found {episode_count} episodes, total {total_frames} frames.")
    except Exception as e:
        print(f"[FAIL] Failed to read meta/episodes.jsonl: {e}")

    # 3. Verify Parquet files
    parquet_files = sorted(glob.glob(os.path.join(dataset_dir, "data/chunk-000/episode_*.parquet")))
    print(f"\nFound {len(parquet_files)} Parquet files.")
    
    if len(parquet_files) > 0:
        try:
            # Read first Parquet schema
            table = pq.read_table(parquet_files[0])
            print(f"[OK] Parquet schema of {os.path.basename(parquet_files[0])}:")
            for field in table.schema:
                print(f"     - {field.name}: {field.type}")
            
            # Print sample values
            print("\nSample values from first step:")
            df = table.to_pandas()
            for col in df.columns:
                print(f"     * {col}: {df[col].iloc[0]}")
        except Exception as e:
            print(f"[FAIL] Failed to read Parquet files: {e}")

    # 4. Verify Videos
    video_files = sorted(glob.glob(os.path.join(dataset_dir, "videos/chunk-000/observation.images.wrist/episode_*.mp4")))
    print(f"\nFound {len(video_files)} Video files.")

    if len(video_files) > 0:
        if HAS_CV2:
            try:
                cap = cv2.VideoCapture(video_files[0])
                if not cap.isOpened():
                    print(f"[FAIL] OpenCV could not open video file: {video_files[0]}")
                else:
                    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    print(f"[OK] Video parameters of {os.path.basename(video_files[0])}:")
                    print(f"     - Resolution: {int(w)}x{int(h)}")
                    print(f"     - FPS: {fps}")
                    print(f"     - Frame count: {int(frame_count)}")
                cap.release()
            except Exception as e:
                print(f"[FAIL] Error reading video via OpenCV: {e}")
        else:
            print("[NOTE] OpenCV not installed, skipping video properties validation.")

    print("\n==================== Verification Finished ====================\n")


if __name__ == "__main__":
    main()
