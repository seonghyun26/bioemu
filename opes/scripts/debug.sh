cd ../

method="debug"
date="debug"
echo $date

python main.py \
    --config-name $method \
    date=$date \
    +tags=['debug'] \
    +opes.ntomp=2 \
    opes.max_seed=0 \
    start_gpu=0

wait 

python analysis_opes.py \
    --config-name $method \
    date=$date \
    opes.max_seed=0
