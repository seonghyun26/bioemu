cd ../

method=ours
num_samples=100
mlcv_dim=2
date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --sequence YYDPETGTWY \
    --score_model_mode eval \
    --condition_mode input \
    --last_training 1 \
    --mlcv_dim $mlcv_dim \
    --time_lag 100 \
    --date $date \
    --param_watch True \
    --eta_max 1e-3 \
    --tags pilot ca_distance_loss larger_time_lag


# CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample-cond \
#     --sequence YYDPETGTWY \
#     --num_samples $num_samples \
#     --output_dir ./res/cln025-$method-$date \
#     --filter_samples True \
#     --mlcv_dim $mlcv_dim \
#     --method $method \
#     --date $date \
#     --condition_mode input

# CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
#     --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$date/topology.pdb \
#     --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$date/samples.xtc \
#     --outpath ./res/cln025-$method-$date

# CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
#     --method $method \
#     --date $date