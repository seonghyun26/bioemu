cd ../

GPU=${1:-4}
method=$2
sigma=$3
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method \
    step=600_000_000 \
    date=$date \
    sigma=$sigma \
    +gpu=$GPU  \
    +background=False

python analysis_opes.py \
    --config-name $method \
    date=$date

