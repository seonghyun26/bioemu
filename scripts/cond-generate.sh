method=ours
date=0729_091523
num_samples=10
condition_mode=input
mlcv_dim=2

cd ../

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample-cond 

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/samples.xtc \
    --pdb-path /home/shpark/prj-mlcv/lib/DESRES/data/CLN025_desres_backbone.pdb \
    --outpath ./res/cln025/$date

CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
    --sample_path ./res/cln025/$date \
    --date $date
