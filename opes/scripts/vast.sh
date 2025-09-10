cd ../

method=$1
molecule=$2
date=$(date +%m%d_%H%M%S)
echo $date


python main.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['vast'] \
    opes.ntomp=$3

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['vast']

