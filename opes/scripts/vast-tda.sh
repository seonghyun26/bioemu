cd ../

method=tda
molecule=${1:-"cln025"}
start_gpu=${2:-"0"}
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method-$molecule \
    date=$date \
    start_gpu=$start_gpu \
    +tags=['vast']

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['vast']

