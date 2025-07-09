method=ours
date=$2
num_samples=100
cfg_dir=cfg4

cd ../

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample-cfg \
    --sequence YYDPETGTWY \
    --num_samples $num_samples \
    --output_dir ./res/cln025-$method-$date-$cfg_dir \
    --filter_samples True \
    --mlcv_dim 2 \
    --method $method \
    --date $date \
    --condition_mode input

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
    --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$date-$cfg_dir/topology.pdb \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$date-$cfg_dir/samples.xtc \
    --outpath ./res/cln025-$method-$date-$cfg_dir

CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
    --method $method \
    --date $date
