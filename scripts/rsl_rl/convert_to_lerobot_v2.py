# Copyright (c) 2026

"""Script to convert raw collected trajectory data to LeRobot v2 format.
   Compiles PNG images to MP4 videos and writes CSV states/actions to Parquet files.
"""

import argparse
import os
import glob
import json
import csv
import cv2
import shutil
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Convert raw Isaac Sim collected data to LeRobot v2.")
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw collected episodes directory.")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to write the LeRobot v2 dataset.")
    parser.add_argument("--fps", type=int, default=10, help="Framerate of the collected trajectory.")
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    out_dir = os.path.abspath(args.out_dir)

    print(f"[INFO] Raw Dir: {raw_dir}")
    print(f"[INFO] Output Dir: {out_dir}")

    # Create directory structure
    meta_dir = os.path.join(out_dir, "meta")
    data_dir = os.path.join(out_dir, "data", "chunk-000")
    video_dir = os.path.join(out_dir, "videos", "chunk-000", "observation.images.wrist")

    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Find raw episodes
    episode_paths = sorted(glob.glob(os.path.join(raw_dir, "episode_*")))
    if len(episode_paths) == 0:
        print(f"[ERROR] No raw episodes found in {raw_dir}")
        return

    print(f"[INFO] Found {len(episode_paths)} raw episodes to process.")

    tasks = []
    task_to_idx = {}
    episodes_meta = []
    global_frame_index = 0

    # Schema for Parquet
    schema = pa.schema([
        ('index', pa.int64()),
        ('episode_index', pa.int64()),
        ('timestamp', pa.float32()),
        ('task_index', pa.int64()),
        ('observation.state', pa.list_(pa.float32(), 9)),
        ('action', pa.list_(pa.float32(), 6)),
        ('next.reward', pa.float32()),
        ('next.done', pa.bool_())
    ])

    for ep_path in episode_paths:
        ep_name = os.path.basename(ep_path)
        ep_idx = int(ep_name.split("_")[1])

        metadata_json_path = os.path.join(ep_path, "metadata.json")
        data_csv_path = os.path.join(ep_path, "data.csv")
        images_path = os.path.join(ep_path, "images")

        if not (os.path.exists(metadata_json_path) and os.path.exists(data_csv_path) and os.path.exists(images_path)):
            print(f"[WARNING] Skipping incomplete episode: {ep_name}")
            continue

        # 1. Read metadata
        with open(metadata_json_path, "r") as f:
            meta = json.load(f)

        task_desc = meta["task_description"]
        if task_desc not in task_to_idx:
            task_idx = len(tasks)
            task_to_idx[task_desc] = task_idx
            tasks.append({"task_index": task_idx, "task": task_desc})
        else:
            task_idx = task_to_idx[task_desc]

        # 2. Compile video from PNG images using OpenCV
        video_output_path = os.path.join(video_dir, f"episode_{ep_idx:06d}.mp4")
        print(f"[OpenCV] Compiling video for episode {ep_idx}...")
        
        # Determine the number of png files to check if it matches CSV rows
        png_files = sorted(glob.glob(os.path.join(images_path, "frame_*.png")))
        num_frames = len(png_files)

        if num_frames > 0:
            # Read first image to determine width and height
            temp_img = cv2.imread(png_files[0])
            h, w, c = temp_img.shape
            
            # Using mp4v codec for writing MP4 video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_output_path, fourcc, args.fps, (w, h))
            
            for png_file in png_files:
                img = cv2.imread(png_file)
                video_writer.write(img)
            video_writer.release()
        else:
            print(f"[WARNING] No frames found in {images_path}")

        # 3. Read data.csv and write to Parquet
        pylist = []
        with open(data_csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            
            rows = list(reader)
            num_rows = len(rows)
            
            if num_rows != num_frames:
                print(f"[WARNING] Row count ({num_rows}) in data.csv does not match frame count ({num_frames}) in images/")

            for r_idx, row in enumerate(rows):
                step = int(row[0])
                joint_pos = [float(x) for x in row[1:7]]
                joint_target = [float(x) for x in row[7:13]]
                
                # Retrieve Goal relative to gripper (cols 13-15) if present, fallback to [0,0,0]
                if len(row) >= 16:
                    goal_rel_ee = [float(x) for x in row[13:16]]
                else:
                    goal_rel_ee = [0.0, 0.0, 0.0]
                
                # Combine joints and relative goal vector into 9D state observation
                combined_state = joint_pos + goal_rel_ee
                
                is_last_step = (r_idx == num_rows - 1)
                
                pylist.append({
                    "index": global_frame_index,
                    "episode_index": ep_idx,
                    "timestamp": float(step) / float(args.fps),
                    "task_index": task_idx,
                    "observation.state": combined_state,
                    "action": joint_target,
                    "next.reward": 0.0,
                    "next.done": is_last_step
                })
                global_frame_index += 1

        # Save Parquet file for this episode
        parquet_table = pa.Table.from_pylist(pylist, schema=schema)
        parquet_output_path = os.path.join(data_dir, f"episode_{ep_idx:06d}.parquet")
        pq.write_table(parquet_table, parquet_output_path)
        print(f"[Parquet] Saved episode Parquet to {parquet_output_path}")

        # Add to episodes metadata list
        episodes_meta.append({
            "episode_index": ep_idx,
            "tasks": [task_idx],
            "length": num_rows
        })

    # 4. Write Metadata files to meta/
    
    # meta/tasks.jsonl
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")
    with open(tasks_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"[Meta] Saved {tasks_path}")

    # meta/episodes.jsonl
    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    with open(episodes_path, "w") as f:
        for e in episodes_meta:
            f.write(json.dumps(e) + "\n")
    print(f"[Meta] Saved {episodes_path}")

    # meta/info.json
    info_json = {
        "codebase_version": "v2.0",
        "fps": args.fps,
        "features": {
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [3, 480, 640],
                "names": ["channels", "height", "width"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [9],
                "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "r_joint", "goal_rel_dx", "goal_rel_dy", "goal_rel_dz"]
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": ["joint1_target", "joint2_target", "joint3_target", "joint4_target", "joint5_target", "r_joint_target"]
            }
        }
    }
    
    info_path = os.path.join(meta_dir, "info.json")
    with open(info_path, "w") as f:
        json.dump(info_json, f, indent=4)
    print(f"[Meta] Saved {info_path}")

    # Copy modality.json if present
    script_dir = os.path.dirname(os.path.abspath(__file__))
    groot_modality_path = os.path.join(script_dir, "groot", "modality.json")
    if os.path.exists(groot_modality_path):
        shutil.copy(groot_modality_path, os.path.join(meta_dir, "modality.json"))
        print(f"[Meta] Copied modality.json to {meta_dir}")

    print("\n[CONVERSION COMPLETED] LeRobot v2 dataset generated successfully!")


if __name__ == "__main__":
    main()
