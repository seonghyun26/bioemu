method=ours
date=$2
num_samples=100

cd ../

# CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample-cond \
#     --sequence YYDPETGTWY \
#     --num_samples $num_samples \
#     --output_dir ./res/cln025/$date \
#     --filter_samples True \
#     --mlcv_dim 2 \
#     --method $method \
#     --date $date \
#     --condition_mode backbone \
#     --ckpt_path /home/shpark/prj-mlcv/lib/bioemu/model/0722_213445/ckpt_100.pt \
#     --model_config_path /home/shpark/prj-mlcv/lib/bioemu/config/model_config.yaml

# CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
#     --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/topology.pdb \
#     --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/samples.xtc \
#     --outpath ./res/cln025/$date

CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
    --method $method \
    --date $date
