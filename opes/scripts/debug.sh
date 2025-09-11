cd ../

method="tica"
ntomp=${2:-"1"}
start_gpu=${3:-"0"}
date="debug"
echo $date

python main.py \
    --config-name debug \
    date=$date \
    +tags=['debug'] \
    start_gpu=$start_gpu \
    opes.ntomp=$ntomp

wait 

python analysis_opes.py \
    --config-name debug \
    date=$date \
    +tags=['debug'] \
    opes.max_seed=0 \
    analysis.skip_steps=0 \
    analysis.unit_steps=1
