cd ../

method="tica"
molecule=${1:-"1fme"}
date="debug"
echo $date

# python main.py \
#     --config-name $method-$molecule \
#     date=$date \
#     +tags=['debug'] \
#     opes.max_seed=0 \
#     opes.step=10000 \
#     start_gpu=0

# wait 

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['debug'] \
    opes.max_seed=0 \
    opes.step=10000 \
    analysis.skip_steps=0 \
    analysis.unit_steps=1
