cd ../

date=$(date +%m%d_%H%M%S)


CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-cln025-coord \
    ++log.date=$date \
    ++log.tags=['pilot','score','coordinate input']
