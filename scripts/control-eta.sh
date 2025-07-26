cd ../

# Define GPU array
gpus=(3 4 5 6 7 8)
eta_max=(1e-3 1e-4 1e-5 1e-6 1e-7 1e-8)
time_lag=(5 10 20 50 100 200)
mlcv_dim=(1 2 4 8 16 32)

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
        --mlcv_dim 5 \
        --time_lag 5 \
        --param_watch True \
        --eta_max $eta_max \
        --learning_rate 1e-10 \
        --tags pilot ca_distance_loss larger_time_lag no_physical_val_loss &

    sleep 2
done

wait

for i in "${!time_lag[@]}"; do
    time_lag=${time_lag[$i]}
    gpu_idx=$((i % ${#gpus[@]}))
    gpu=${gpus[$gpu_idx]}
    
    echo "Running experiment with time_lag=$time_lag on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python control.py \
        --sequence YYDPETGTWY \
        --score_model_mode eval \
        --condition_mode input \
        --last_training 1 \
        --mlcv_dim 5 \
        --time_lag $time_lag \
        --param_watch True \
        --eta_max 1e-4 \
        --learning_rate 1e-10 \
        --tags pilot ca_distance_loss larger_time_lag no_physical_val_loss &

    sleep 2
done

wait

for i in "${!mlcv_dim[@]}"; do
    mlcv_dim=${mlcv_dim[$i]}
    gpu_idx=$((i % ${#gpus[@]}))
    gpu=${gpus[$gpu_idx]}
    
    echo "Running experiment with mlcv_dim=$mlcv_dim on GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu python control.py \
        --sequence YYDPETGTWY \
        --score_model_mode eval \
        --condition_mode input \
        --last_training 1 \
        --mlcv_dim $mlcv_dim \
        --time_lag 100 \
        --param_watch True \
        --eta_max 1e-4 \
        --learning_rate 1e-10 \
        --tags pilot ca_distance_loss larger_time_lag no_physical_val_loss &

    sleep 2
done

wait


