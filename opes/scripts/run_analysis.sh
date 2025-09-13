cd ../

method=${1:-"tda"}
molecule=${2:-"cln025"}
date=${3:-$(date +%m%d_%H%M%S)}
echo $date

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    ++analysis.gmx=True \
    +tags=['vast']

