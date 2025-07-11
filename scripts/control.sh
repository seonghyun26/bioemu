cd ../

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --sequence YYDPETGTWY \
    --score_model_mode eval \
    --condition_type input \
    --last_training 1 \
    --mlcv_dim 2 \
    --time_lag 5 \
    --tags pilot