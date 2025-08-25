cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-transferable \
    ++log.date=$date \
    ++log.tags=['score','transferable'] 