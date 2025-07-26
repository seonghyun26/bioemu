cd ../

method=ours
num_samples=100
mlcv_dim=2
date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --sequence YYDPETGTWY \
    --score_model_mode eval \
    --condition_mode backbone \
    --last_training 1 \
    --mlcv_dim $mlcv_dim \
    --time_lag 1 \
    --param_watch True \
    --eta_max 1e-6 \
    --learning_rate 1e-12 \
    --tags pilot reconstruction