method=$2
date=$(date +"%m%d-%H%M")
num_samples=100

cd ../

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample \
    --sequence YYDPETGTWY \
    --num_samples $num_samples \
    --output_dir ./res/cln025-$method-$date \
    --filter_samples True 

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
    --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$date/topology.pdb \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$date/samples.xtc \
    --outpath ./res/cln025-$method-$date