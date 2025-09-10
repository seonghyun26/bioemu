cd ../

method=ours
molecule=${1:-"2jof"}
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr'] \
    start_gpu=0

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr']
