cd ../

method=ours
molecule=cln025
date=$(date +%m%d_%H%M%S)
date="0830_162102"
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
