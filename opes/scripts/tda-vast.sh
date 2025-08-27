cd ../

GPU=${1:-4}
sigma=$2
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name tda \
    step=600_000_000 \
    date=$date \
    sigma=$sigma \
    gpu=$GPU 

python analysis_opes.py \
        --config-name tda \
        date=$date

