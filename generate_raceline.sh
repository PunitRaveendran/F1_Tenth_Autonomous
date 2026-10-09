#!/bin/bash

# F1TENTH Raceline Generation Pipeline
# Automatically extracts centerline, optimizes trajectory, and generates speed profile.

if [ -z "$1" ]; then
    echo "Usage: ./generate_raceline.sh <map_name>"
    echo "Example: ./generate_raceline.sh Spielberg_map"
    echo "Note: Do not include the .png or .yaml extensions in the map name."
    exit 1
fi

MAP_NAME=$1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}"
MAP_DIR="${WORKSPACE_DIR}/src/f1tenth_gym_ros/maps"

echo "======================================================="
echo " Starting Raceline Generation for map: ${MAP_NAME}"
echo "======================================================="

# Check if map exists
if [ ! -f "${MAP_DIR}/${MAP_NAME}.png" ] || [ ! -f "${MAP_DIR}/${MAP_NAME}.yaml" ]; then
    echo "Error: Map files not found!"
    echo "Expected: ${MAP_DIR}/${MAP_NAME}.png and .yaml"
    exit 1
fi

cd ${WORKSPACE_DIR}

echo "[1/4] Extracting Centerline..."
python3 extract_centerline.py --map_name ${MAP_NAME} --map_dir ${MAP_DIR}

if [ $? -ne 0 ]; then
    echo "Error during centerline extraction."
    exit 1
fi

echo "[2/4] Running Global Trajectory Optimization..."
# Copy the centerline to the optimizer's input directory
CENTERLINE_CSV="${MAP_DIR}/${MAP_NAME}_centerline.csv"
OPT_INPUT="${WORKSPACE_DIR}/global_racetrajectory_optimization/inputs/tracks/f1tenth_track.csv"

cp ${CENTERLINE_CSV} ${OPT_INPUT}
# The optimizer requires the first column header to be commented out
sed -i '1s/^x_m/# x_m/' ${OPT_INPUT}

cd ${WORKSPACE_DIR}/global_racetrajectory_optimization
python3 main_globaltraj.py

if [ $? -ne 0 ]; then
    echo "Error during global trajectory optimization."
    exit 1
fi

echo "[3/4] Generating Speed Profile..."
cd ${WORKSPACE_DIR}
python3 speed_profile.py --map_name ${MAP_NAME} --map_dir ${MAP_DIR}

if [ $? -ne 0 ]; then
    echo "Error during speed profile generation."
    exit 1
fi

echo "[4/4] Generating Visualization..."
cd ${WORKSPACE_DIR}
python3 visualize_raceline.py --map_name ${MAP_NAME} --map_dir ${MAP_DIR}

echo "======================================================="
echo " Pipeline Complete!"
echo " Centerline saved to: ${MAP_DIR}/${MAP_NAME}_centerline.csv"
echo " Raceline saved to:   ${MAP_DIR}/${MAP_NAME}_raceline.csv"
echo " Visualization:       ${WORKSPACE_DIR}/raceline_visualization.png"
echo " Default raceline.csv updated in maps directory."
echo "======================================================="
