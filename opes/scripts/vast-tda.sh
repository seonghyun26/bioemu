cd ../

GPU=${1:-0}
sigma=${2:-0.1}
method=tda
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method \
    step=200_000_000 \
    date=$date \
    sigma=$sigma \
    gpu=$GPU 

python analysis_opes.py \
    --config-name $method \
    date=$date \
    sigma=$sigma

