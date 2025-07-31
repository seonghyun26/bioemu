method=ours
date=0729_081523
num_samples=100
condition_mode=input
mlcv_dim=2

cd ../

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample-cond \
    --sequence YYDPETGTWY \
    --num_samples $num_samples \
    --output_dir ./res/cln025/$date \
    --filter_samples True \
    --mlcv_dim $mlcv_dim \
    --method $method \
    --date $date \
    --condition_mode $condition_mode \
    --ckpt_path /home/shpark/prj-mlcv/lib/bioemu/model/0722_213445/ckpt_100.pt \
    --model_config_path /home/shpark/prj-mlcv/lib/bioemu/config/model_config.yaml

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/samples.xtc \
    --pdb-path /home/shpark/prj-mlcv/lib/DESRES/data/CLN025_desres_backbone.pdb \
    --outpath ./res/cln025/$date
    # --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/topology.pdb \

CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
    --sample_path ./res/cln025/$date \
    --date $date
