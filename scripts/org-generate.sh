method=original
date=$(date +"%m%d_%H%M")
# date=0726_152918
num_samples=10000
echo $date

cd ../

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample \
    --sequence YYDPETGTWY \
    --num_samples $num_samples \
    --output_dir ./res/cln025-org/$date \
    --filter_samples True 

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
    --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-org/$date/topology.pdb \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-org/$date/samples.xtc \
    --outpath ./res/cln025-org/$date

CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
    --sample_path ./res/cln025-org/$date \
    --date $date