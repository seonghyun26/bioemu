cd ../

method=ours
molecule=${1:-"2jof"}
start_gpu=${2:-0}
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr'] \
    start_gpu=$start_gpu

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr']
