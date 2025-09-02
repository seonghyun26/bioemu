method=original
date=$(date +"%m%d_%H%M")
# date=0726_152918
num_samples=1000
molecule=2JOF
echo $date

cd ../

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample \
    --sequence DAYAQWLADGGPSSGRPPPS \
    --num_samples $num_samples \
    --output_dir ./res/$molecule-org/$date \
    --filter_samples False 

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
    --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/$molecule-org/$date/topology.pdb \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/$molecule-org/$date/samples.xtc \
    --outpath ./res/$molecule-org/$date