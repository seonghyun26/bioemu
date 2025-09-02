method=ours
date=0804_062229
num_samples=100
condition_mode=input
mlcv_dim=1

cd ../

for bond in 0 1 2 3 4 5 6 7; do
    echo "Sampling with condition $bond bond..."

    CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sample-cond \
        ++sample.output_dir=./res/cln025/$date/$bond"bond" \
        ++sample.cond_pdb=/home/shpark/prj-mlcv/lib/DESRES/data/$bond"bond.pdb"
    sleep 1

    CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
        --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/$bond"bond"/samples.xtc \
        --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025/$date/topology.pdb \
        --outpath ./res/cln025/$date/$bond"bond"
    sleep 1

    CUDA_VISIBLE_DEVICES=$1 python classify_samples.py \
        --sample_path ./res/cln025/$date/$bond"bond" \
        --date $date
    sleep 1
done