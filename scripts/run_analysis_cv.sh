#!/bin/bash

# Run CV analysis script for different model types
# Usage: ./run_analysis_cv.sh [model_type] [molecule] [date]
# 
# model_type: mlcv, tae, vde, or all (default: all)
# molecule: CLN025 or 2JOF (default: CLN025)
# date: date string for MLCV model (optional, only used for mlcv)

set -e  # Exit on any error

# Default values
MODEL_TYPE=${2:-all}
MOLECULE=${3:-CLN025}
DATE=${4:-}

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMG_DIR="$PROJECT_ROOT/img"

# Create img directory if it doesn't exist
mkdir -p "$IMG_DIR"

# Python script path
PYTHON_SCRIPT="$PROJECT_ROOT/analysis_cv.py"

echo "========================================"
echo "Running CV Analysis"
echo "========================================"
echo "Model type: $MODEL_TYPE"
echo "Molecule: $MOLECULE"
echo "Date: ${DATE:-'default'}"
echo "Image directory: $IMG_DIR"
echo "========================================"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Build command
CMD="CUDA_VISIBLE_DEVICES=$1 python3 $PYTHON_SCRIPT --model_type $MODEL_TYPE --molecule $MOLECULE --img_dir $IMG_DIR --plot_3d True"

# Add date parameter if provided
if [ -n "$DATE" ]; then
    CMD="$CMD --date $DATE"
fi

echo "Running command: $CMD"
echo "========================================"

# Run the analysis
cd "$PROJECT_ROOT"
eval "$CMD"

echo "========================================"
echo "Analysis completed successfully!"
echo "Results saved to: $IMG_DIR"
echo "========================================"

# List generated files
# echo "Generated files:"
# ls -la "$IMG_DIR"/*.png 2>/dev/null || echo "No PNG files found in $IMG_DIR"
