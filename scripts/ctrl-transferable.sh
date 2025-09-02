cd ../

date=$(date +%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=$1 python control.py \
    --config-name score-trans2 \
    ++log.date=$date \
    ++log.tags=['score','trans2','residue encoding'] 