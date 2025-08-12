cd ../

date=$(date +%m%d_%H%M%S)


CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-2jof \
    ++log.date=$date \
    model.training.learning_rate=1e-12 \
    model.training.scheduler.eta_max=1e-6 \
    model.training.num_epochs=800 \
    model.training.batch_size=512 \
    ++log.tags=['pilot','score']
