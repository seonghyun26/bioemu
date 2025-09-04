cd ../

method=ours
molecule=2jof
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method-$molecule \
    opes.step=500_000_000 \
    date=$date \
    ++start_gpu=4 \
    +tags=['svr']

python analysis_opes.py \
    --config-name $method \
    date=$date \
    +tags=['svr']
