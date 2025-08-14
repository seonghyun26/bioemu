cd ../

date=$(date +%m%d_%H%M%S)


CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score \
    ++log.date=$date \
    model.training.num_epochs=800 \
    ++log.tags=['pilot','score']
