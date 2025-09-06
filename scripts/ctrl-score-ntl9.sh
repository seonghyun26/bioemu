cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-ntl9 \
    ++model.training.batch_size=200 \
    ++model.training.num_epochs=800 \
    ++log.date=$date \
    ++log.tags=['score']
