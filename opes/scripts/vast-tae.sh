cd ../

method=tae
molecule=${1:-"cln025"}
date=$(date +%m%d_%H%M%S)
echo $date

python main.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['vast'] \
    opes.ntomp=$2

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['vast']

