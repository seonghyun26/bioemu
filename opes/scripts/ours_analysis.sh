cd ../

method=ours
molecule=cln025
date=0907_065813
echo $date


python analysis_opes.py \
    --config-name $method-$molecule \
    date=$date \
    +tags=['svr']
