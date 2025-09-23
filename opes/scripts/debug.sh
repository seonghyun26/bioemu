cd ../

method=${1:-"tica"}
molecule=${2:-"1fme"}
ntomp=${3:-"1"}
start_gpu=${4:-"0"}
max_seed=${5:-"3"}
date="debug"
echo $date

python main.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['debug'] \
    opes.step=10_000 \
    opes.max_seed=$max_seed \
    start_gpu=$start_gpu \
    opes.ntomp=$ntomp

wait 

# python analysis_opes.py \
#     --config-name $method-$molecule \
#     date=$date \
#     +tags=['debug'] \
#     opes.max_seed=$max_seed \
#     analysis.gmx=False \
#     analysis.skip_steps=10 \
#     analysis.unit_steps=10
