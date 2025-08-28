cd ../

GPU=${1:-7}
SIGMA=${2:-0.2}
method="ours"
date="0826_062345"
echo $date

# python main.py \
#     --config-name $method \
#     step=10_000 \
#     date=$date \
#     gpu=$GPU \
#     sigma=$SIGMA \

python analysis_opes.py \
    --config-name $method \
    date=$date
