method=ours
date=0729_091523
num_samples=100
condition_mode=input
mlcv_dim=2

cd ../

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample-cond \
    ++sample.output_dir=./res/cln025/$date/1bond \
    ++sample.cond_pdb=/home/shpark/prj-mlcv/lib/DESRES/data/1bond.pdb

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/1bond/samples.xtc \
    --pdb-path /home/shpark/prj-mlcv/lib/DESRES/data/CLN025_desres_backbone.pdb \
    --outpath ./res/cln025/$date/1bond

CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
    --sample_path ./res/cln025/$date/1bond \
    --date $date
