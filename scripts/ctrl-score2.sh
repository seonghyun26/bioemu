cd ../

date=$(date +%m%d_%H%M%S)
learning_rate=1e-8

# CUDA_VISIBLE_DEVICES=$1 python control.py \
#     --config-name score \
#     ++log.date=$date \
#     ++log.tags=['pilot','score']

for eta_max in 1e-4 1e-6 1e-8 1e-10; do
    CUDA_VISIBLE_DEVICES=$1 python control.py \
        --config-name score \
        ++log.date=$date \
        model.training.learning_rate=$learning_rate \
        model.training.scheduler.eta_max=$eta_max \
        ++log.tags=['pilot','score']
done

for eta_max in 1e-4 1e-6 1e-8 1e-10; do
    CUDA_VISIBLE_DEVICES=$1 python control.py \
        --config-name score \
        ++log.date=$date \
        model.training.learning_rate=$learning_rate \
        model.training.scheduler.eta_max=$eta_max \
        model.training.num_epochs=1000 \
        ++log.tags=['pilot','score']
done
