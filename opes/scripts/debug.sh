cd ../

method="debug"
date="debug"
echo $date

python main.py \
    --config-name $method \
    date=$date \
    +tags=['debug'] \
    opes.max_seed=0 \
    start_gpu=1

wait 

# python analysis_opes.py \
#     --config-name $method \
#     date=$date \
#     opes.max_seed=0
