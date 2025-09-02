cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-2jof \
    model.training.num_epochs=800 \
    model.training.batch_size=512 \
    model.mlcv_model.mlcv_dim=2 \
    data.dataset_size=50k \
    ++log.date=$date \
    ++log.tags=['pilot','score','2dim']
