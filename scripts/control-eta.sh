cd ../

# Define GPU array
gpus=(1 2)
eta_max=(1e-2 1e-3)

# Run experiments on different GPUs
for i in "${!eta_max[@]}"; do
    eta_max=${eta_max[$i]}
    gpu_idx=$((i % ${#gpus[@]}))
    gpu=${gpus[$gpu_idx]}
    
    echo "Running experiment with eta_max=$eta_max on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python control.py \
        --sequence YYDPETGTWY \
        --score_model_mode eval \
        --condition_mode input \
        --last_training 1 \
        --mlcv_dim 2 \
        --time_lag 5 \
        --tags learning_rate_abl train_mlp \
        --eta_max $eta_max &

    sleep 1
done

# Wait for all background processes to complete
wait
echo "All experiments completed!"