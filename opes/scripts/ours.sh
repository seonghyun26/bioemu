cd ../

GPU=${1:-7}
SIGMA=${2:-0.2}
method=ours
date=$(date +%m%d_%H%M%S)
# date="0826_062345"
# date="debug"
echo $date

python main.py \
    --config-name $method \
    step=200_000_000 \
    date=$date \
    gpu=$GPU \
    sigma=$SIGMA \

wait

sleep 1

python analysis_opes.py \
    --config-name $method \
    date=$date \
    sigma=$SIGMA

wait 

sleep 1




date=$(date +%m%d_%H%M%S)
# date="0826_062345"
# date="debug"
echo $date

python main.py \
    --config-name $method \
    step=600_000_000 \
    date=$date \
    gpu=$GPU \
    sigma=$SIGMA \

wait

sleep 1

python analysis_opes.py \
    --config-name $method \
    date=$date \
    sigma=$SIGMA
