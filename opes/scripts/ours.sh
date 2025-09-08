cd ../

method=ours
molecule=${1:-"2jof"}
date=$(date +%m%d_%H%M%S)
date="0906_161400"
echo $date

# python main.py \
#     --config-name $method-$molecule \
#     date=$date \
#     +tags=['svr'] \
#     start_gpu=4 

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr']
