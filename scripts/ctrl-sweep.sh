#!/bin/bash

# Multi-GPU Parameter Sweep Script using Hydra Multirun (Fixed Version)
# Usage: ./ctrl-sweep-fixed.sh [num_gpus] [config_name]
# Example: ./ctrl-sweep-fixed.sh 4 score

cd ../

# Default values
NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}  # Use all available GPUs by default
CONFIG_NAME="score"
base_date=$(date +%m%d_%H%M%S)

echo "Starting parameter sweep with $NUM_GPUS GPUs using config: $CONFIG_NAME"
echo "Sweep started at: $(date)"
echo "Base experiment ID: $base_date"

# Create output directory for this sweep
mkdir -p "sweeps/$base_date"

echo "=== Launching Multi-GPU Parameter Sweep ==="

# Method: Manual GPU assignment with unique log directories
# Each GPU gets its own log.date to prevent conflicts
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    echo "Launching parameter sweep on GPU $gpu_id"
    
    # Create unique date for this GPU
    gpu_date="${base_date}_gpu${gpu_id}"
    
    # Each GPU gets a different subset of parameters
    case $gpu_id in
        0)
            echo "  GPU 0: Testing MLCV dimensions and small learning rates"
            PARAMS="model.training.learning_rate=1e-6 model.training.scheduler.eta_max=1e-4,1e-6,1e-8,1e-10"
            ;;
        1)
            echo "  GPU 1: Testing larger MLCV dimensions and higher learning rates"
            PARAMS="model.training.learning_rate=1e-8 model.training.scheduler.eta_max=1e-4,1e-6,1e-8,1e-10"
            ;;
        2)
            echo "  GPU 2: Testing batch sizes and rollout parameters"
            PARAMS="model.training.learning_rate=1e-10 model.training.scheduler.eta_max=1e-4,1e-6,1e-8,1e-10"
            ;;
        3)
            echo "  GPU 3: Testing conditioning modes and training methods"
            PARAMS="model.training.learning_rate=1e-12 model.training.scheduler.eta_max=1e-4,1e-6,1e-8,1e-10"
            ;;
        *)
            # For additional GPUs, cycle through parameter combinations
            # idx=$((gpu_id % 4))
            # case $idx in
            #     0) 
            #         echo "  GPU $gpu_id: Testing MLCV dimensions (cycle)"
            #         PARAMS="model.mlcv_model.mlcv_dim=1,2,4 model.training.learning_rate=1e-6,1e-5" 
            #         ;;
            #     1) 
            #         echo "  GPU $gpu_id: Testing larger dimensions (cycle)"
            #         PARAMS="model.mlcv_model.mlcv_dim=8,16 model.training.learning_rate=1e-4,1e-3" 
            #         ;;
            #     2) 
            #         echo "  GPU $gpu_id: Testing batch sizes (cycle)"
            #         PARAMS="model.training.batch_size=32,64,128 model.rollout.mid_t=0.5,0.7,0.8" 
            #         ;;
            #     3) 
            #         echo "  GPU $gpu_id: Testing conditioning modes (cycle)"
            #         PARAMS="model.mlcv_model.condition_mode=input,latent model.training.method=standard,ppft" 
            #         ;;
            # esac
            # ;;
    esac
    
    # Launch job on specific GPU with UNIQUE log.date
    CUDA_VISIBLE_DEVICES=$gpu_id python control.py \
        --config-name $CONFIG_NAME \
        --multirun \
        hydra.mode=MULTIRUN \
        hydra.sweep.dir="sweeps/$base_date/gpu_$gpu_id" \
        hydra.job.chdir=True \
        ++log.date="$gpu_date" \
        ++log.tags="['sweep','gpu$gpu_id','$CONFIG_NAME','$base_date']" \
        $PARAMS > "sweeps/$base_date/gpu_${gpu_id}_output.log" 2>&1 &
    
    # Store the process ID for monitoring
    echo $! > "sweeps/$base_date/gpu_${gpu_id}.pid"
    echo "  Started on GPU $gpu_id with PID: $(cat sweeps/$base_date/gpu_${gpu_id}.pid)"
    
    # Small delay to prevent resource conflicts
    sleep 3
done

echo ""
echo "=== All parameter sweep jobs launched! ==="
echo "Base experiment ID: $base_date"
echo "Individual GPU experiments:"
for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
    gpu_date="${base_date}_gpu${gpu_id}"
    echo "  GPU $gpu_id: log.date='$gpu_date'"
done

echo ""
echo "=== Monitoring Commands ==="
echo "Monitor GPU usage:     watch -n 5 nvidia-smi"
echo "Monitor processes:     watch -n 5 'ps aux | grep python | grep control.py'"
echo "Monitor logs:          tail -f sweeps/$base_date/gpu_*_output.log"
echo "Check progress:        ls -la sweeps/$base_date/"

echo ""
echo "=== Results will be saved in: ==="
echo "Main directory:        sweeps/$base_date/"
echo "Per-GPU directories:   sweeps/$base_date/gpu_*/"
echo "Individual logs:       sweeps/$base_date/gpu_*_output.log"

# Function to check if jobs are still running
check_jobs() {
    echo ""
    echo "=== Job Status Check ==="
    for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
        if [ -f "sweeps/$base_date/gpu_${gpu_id}.pid" ]; then
            pid=$(cat "sweeps/$base_date/gpu_${gpu_id}.pid")
            if ps -p $pid > /dev/null 2>&1; then
                echo "GPU $gpu_id (PID: $pid): RUNNING"
            else
                echo "GPU $gpu_id (PID: $pid): FINISHED"
            fi
        fi
    done
}

# Optional: Wait for all background jobs to complete
read -p "Wait for all jobs to complete? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Waiting for all jobs to complete..."
    echo "You can press Ctrl+C to stop waiting and let jobs run in background"
    
    # Check status every 30 seconds
    while true; do
        check_jobs
        
        # Check if any jobs are still running
        any_running=false
        for gpu_id in $(seq 0 $((NUM_GPUS-1))); do
            if [ -f "sweeps/$base_date/gpu_${gpu_id}.pid" ]; then
                pid=$(cat "sweeps/$base_date/gpu_${gpu_id}.pid")
                if ps -p $pid > /dev/null 2>&1; then
                    any_running=true
                    break
                fi
            fi
        done
        
        if [ "$any_running" = false ]; then
            echo "All jobs completed!"
            break
        fi
        
        echo "Some jobs still running... checking again in 30 seconds"
        sleep 30
    done
    
    wait  # Wait for all background processes
    
    echo ""
    echo "=== Parameter sweep completed at: $(date) ==="
    check_jobs
else
    echo "Jobs are running in the background."
    echo "Use the monitoring commands above to check progress."
fi

echo ""
echo "=== Final Results Location ==="
echo "Results saved in: sweeps/$base_date/"
echo "To analyze results:"
echo "  cd sweeps/$base_date"
echo "  find . -name '*.log' | head -10  # Check log files"
echo "  find . -name 'config.yaml' | head -5  # Check configurations"
