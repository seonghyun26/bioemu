cd ../

GPU=${1:-7}
sigma=${2:-0.1}
method=tica
date=$(date +%m%d_%H%M%S)
date="0829_083600"
echo $date

# python main.py \
#     --config-name $method \
#     step=200_000_000 \
#     date=$date \
#     sigma=$sigma \
#     gpu=$GPU &

wait 

sleep 1

python analysis_opes.py \
    --config-name $method \
    date=$date \
    sigma=$sigma
