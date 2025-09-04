cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-gtt \
    ++model.training.batch_size=256 \
    ++log.date=$date \
    ++log.tags=['pilot','score']
