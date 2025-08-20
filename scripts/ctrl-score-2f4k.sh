cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-2f4k \
    model.training.num_epochs=400 \
    model.training.batch_size=256 \
    model.mlcv_model.mlcv_dim=1 \
    data.dataset_size=50k \
    ++log.date=$date \
    ++log.tags=['pilot','score']
