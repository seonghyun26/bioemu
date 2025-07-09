method=original
date=$(date +"%m%d_%H%M")
date=0708_0806
num_samples=1000

cd ../

# CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample \
#     --sequence YYDPETGTWY \
#     --num_samples $num_samples \
#     --output_dir ./res/cln025-$method-$date \
#     --filter_samples True 

# CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
#     --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$date/topology.pdb \
#     --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$date/samples.xtc \
#     --outpath ./res/cln025-$method-$date

CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
    --method $method \
    --date $date