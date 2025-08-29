cd ../

GPU=${1:-3}
sigma=${2:-0.1}
method=vde
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method \
    step=300_000_000 \
    date=$date \
    sigma=$sigma \
    gpu=$GPU 

python analysis_opes.py \
    --config-name $method \
    date=$date

