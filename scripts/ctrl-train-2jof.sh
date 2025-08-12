cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name pilot-2jof \
    ++log.date=$date \
    ++log.tags=['pilot','1dim mlcv'] \
    ++model.training.batch_size=100 \
    ++model.mlcv_model.mlcv_dim=1 \
    ++model.mlcv_model.dim_normalization=False