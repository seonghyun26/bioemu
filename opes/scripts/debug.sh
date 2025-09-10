cd ../

method="tica"
molecule=${1:-"1fme"}
date="debug"
echo $date

python main.py \
    --config-name debug \
    date=$date \
    +tags=['debug'] \
    opes.max_seed=3 \
    start_gpu=0 \

# wait 

# python analysis_opes.py \
#     --config-name debug \
#     date=$date \
#     +tags=['debug'] \
#     opes.max_seed=0 \
#     analysis.skip_steps=0 \
#     analysis.unit_steps=1
