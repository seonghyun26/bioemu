cd ../

method=tica
molecule=${1:-"cln025"}
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method-$molecule \
    opes.step=400_000_000 \
    date=$date \
    +tags=['vast']

python analysis_opes.py \
    --config-name $method \
    date=$date \
    +tags=['vast']

