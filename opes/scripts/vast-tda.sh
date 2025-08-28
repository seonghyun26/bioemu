cd ../

GPU=${1:-0}
sigma=$2
method=tda
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method \
    step=600_000_000 \
    date=$date \
    sigma=$sigma \
    gpu=$GPU 

python analysis_opes.py \
    --config-name $method \
    date=$date

