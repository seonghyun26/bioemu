cd ../

GPU=${1:-6}
sigma=${2:-0.02}
method=ours
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method \
    step=300_000_000 \
    date=$date \
    sigma=$sigma \
    gpu=$GPU &

wait 

sleep 1

python analysis_opes.py \
    --config-name $method \
    date=$date \
    sigma=$sigma
