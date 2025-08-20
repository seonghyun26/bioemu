cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name pilot \
    ++log.date=$date \
    ++log.tags=['pilot','large mlcv dim'] \
    ++model.mlcv_model.mlcv_dim=2