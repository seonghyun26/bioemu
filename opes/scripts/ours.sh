cd ../

method=ours
molecule=${1:-"cln025"}
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr'] \
    start_gpu=4 

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr']
