cd ../

method="tica"
molecule=${1:-"1fme"}
ntomp=${2:-"1"}
date="debug"
echo $date

python main.py \
    --config-name debug \
    date=$date \
    +tags=['debug'] \
    start_gpu=0 \
    opes.ntomp=$ntomp

wait 

python analysis_opes.py \
    --config-name debug \
    date=$date \
    +tags=['debug'] \
    opes.max_seed=0 \
    analysis.skip_steps=0 \
    analysis.unit_steps=1
