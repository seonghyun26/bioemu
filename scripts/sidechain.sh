method=$2
version=$3

cd ../

CUDA_VISIBLE_DEVICES=$1 python -m bioemu.sidechain_relax \
    --pdb-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$version/topology.pdb \
    --xtc-path /home/shpark/prj-mlcv/lib/bioemu/res/cln025-$method-$version/samples.xtc \
    --outpath ./res/cln025-$method-$version