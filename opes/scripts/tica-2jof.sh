cd ../

GPU=${1:-7}
sigma=${2:-0.05}
method=tica
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method-2jof \
    step=200_000_000 \
    date=$date \
    sigma=$sigma \
    gpu=$GPU &

wait 

sleep 1

python analysis_opes.py \
    --config-name $method-2jof \
    date=$date \
    sigma=$sigma
