cd ../

date=$(date +%m%d_%H%M%S)


CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-cln025 \
    ++log.date=$date \
    ++log.tags=['pilot','score','early stopping']
