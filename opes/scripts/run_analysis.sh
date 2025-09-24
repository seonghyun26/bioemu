cd ../

method=${1:-"tda"}
molecule=${2:-"cln025"}
date=${3:-$(date +%m%d_%H%M%S)}
ckpt_date=${4:-$(3)}
gmx=${4:-"False"}
echo $date

python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    analysis.gmx=$gmx \
    +tags=['vast'] \
    +ckpt_date=$ckpt_date

