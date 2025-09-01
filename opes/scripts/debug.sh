cd ../

GPU=${1:-0}
sigma=${2:-0.1}
method="debug"
date="debug"
echo $date

python main.py \
    --config-name $method \
    step=10_000 \
    date=$date \
    gpu=$GPU \
    sigma=$sigma \

wait 

python analysis_opes.py \
    --config-name $method \
    date=$date \
    sigma=$sigma
