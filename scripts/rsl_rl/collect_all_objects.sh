#!/bin/bash

# Configuration
CHECKPOINT="/workspace/test_isaaclab/pickup_place_direct_0203/logs/rsl_rl/pickup_place_direct_0510/2026-05-23_06-29-56_JetRover_0523_DeltaIK_Run_resume1/model_10150.pt"
RAW_OUT_DIR="/workspace/test_isaaclab/groot_data_raw"
LEROBOT_OUT_DIR="/workspace/test_isaaclab/groot_data_lerobot"
NUM_EPISODES=4

# List of 15 Object IDs
OBJECTS=(22 25 26 27 28 31 39 40 41 62 68 70 93 95 96)

# Corresponding English descriptions for language-conditioned GR00T policy
DESCRIPTIONS=(
    "Pick up the red plastic basin from the floor"
    "Pick up the stainless steel fork from the floor"
    "Pick up the stainless steel spoon from the floor"
    "Pick up the stainless steel knife from the floor"
    "Pick up the black spatula from the floor"
    "Pick up the marker pen from the floor"
    "Pick up the green plastic cup from the floor"
    "Pick up the yellow plastic cup from the floor"
    "Pick up the red plastic cup from the floor"
    "Pick up the white ceramic plate from the floor"
    "Pick up the orange plastic toy from the floor"
    "Pick up the blue plastic bottle from the floor"
    "Pick up the red plastic bottle from the floor"
    "Pick up the glass bottle from the floor"
    "Pick up the glass seasoning jar from the floor"
)

# Clean up raw output dir if starting fresh (optional)
# rm -rf /home/eric/isaaclab_volume/groot_data_raw/*

echo "=== STARTING JETROVER GR00T DATA COLLECTION PIPELINE ==="

# 1. Run Data Collection for each object
for i in "${!OBJECTS[@]}"; do
    OBJ_ID=${OBJECTS[$i]}
    DESC="${DESCRIPTIONS[$i]}"
    
    echo "--------------------------------------------------------"
    echo "Collecting data for Object ID: $OBJ_ID"
    echo "Description: '$DESC'"
    echo "--------------------------------------------------------"
    
    docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
        /workspace/test_isaaclab/pickup_place_direct_0203/scripts/rsl_rl/collect_groot_data_0510.py \
        --checkpoint "$CHECKPOINT" \
        --num_episodes "$NUM_EPISODES" \
        --object_id "$OBJ_ID" \
        --output_dir "$RAW_OUT_DIR" \
        --task_description "$DESC" \
        --headless
done

echo "=== DATA COLLECTION FINISHED ==="
echo "=== CONVERTING RAW DATA TO LEROBOT V2 FORMAT ==="

# 2. Run conversion to LeRobot v2
docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
    /workspace/test_isaaclab/pickup_place_direct_0203/scripts/rsl_rl/convert_to_lerobot_v2.py \
    --raw_dir "$RAW_OUT_DIR" \
    --out_dir "$LEROBOT_OUT_DIR"

echo "=== VALIDATING LEROBOT DATASET ==="

# 3. Validate converted dataset
docker exec isaac-lab-base /workspace/isaaclab/isaaclab.sh -p \
    /workspace/test_isaaclab/pickup_place_direct_0203/scripts/rsl_rl/groot/verify_groot_dataset.py \
    --dataset_dir "$LEROBOT_OUT_DIR"

echo "=== PIPELINE EXECUTION COMPLETED ==="
