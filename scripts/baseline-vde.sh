cd ../

# python -m bioemu.sample-cond \
#     --sequence GYDPETGTWG \
#     --num_samples 20 \
#     --output_dir ./res/cln025-vde \
#     --filter_samples False

python -m bioemu.sidechain_relax \
    --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-vde/topology.pdb \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-vde/samples.xtc \
    --outpath ./res/cln025-vde