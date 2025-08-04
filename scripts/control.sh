cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name input \
    ++log.date=$date \
    ++model.score_model.init=xavier \
    ++log.tags=['pilot','xavier'] \
    ++data.representation=cad \
    ++model.mlcv_model.mlcv_dim=2 
    # ++data.dataset_size=50k
    # ++model.training.scheduler.eta_max=1e-4 \
    # ++model.training.gradient_clip_val=1 \
    # --sequence YYDPETGTWY \
    # --score_model_mode eval \
    # --condition_mode backbone \
    # --last_training 1 \
    # --mlcv_dim $mlcv_dim \
    # --time_lag 10 \
    # --param_watch True \
    # --eta_max 1e-6 \
    # --learning_rate 1e-12 \
    # --tags pilot 


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
#     --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/topology.pdb \
#     --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/samples.xtc \
#     --outpath ./res/cln025/$date

# CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
#     --method $method \
#     --date $date