cd ../

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --sequence YYDPETGTWY \
    --score_model_mode eval \
    --condition_type backbone \
    --last_training 0.9 \
    --mlcv_dim 2 \
    --tags condition_input